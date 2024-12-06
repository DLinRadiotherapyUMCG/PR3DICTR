import wandb
import optuna
import logging

from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset
from src.models.tools.save_model import save_model, save_config, save_dataset, save_dataset_summary
from src.training.train import train, validate
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.utils.loss_func.get_loss_function import get_loss_function
from src.utils.fileHandler import create_file, create_folder, create_textfile
from src.dataset.get_dataloader import make_dataloader   
from src.dataset.get_transforms import get_transforms
from src.training.k_fold_cross_validation import K_fold_cross_validation

import os
import numpy as np

from functools import partial

import src.hyper_opt.Optuna_hpt as Optuna_hpt
import src.hyper_opt.WandB_hpt as WandB_hpt
from src.hyper_opt.hyperHandler import HyperTuning_Handler




class experimentHandler():
    """
    TODO: Add description
    """
    
    def __init__(self, config) -> None:
        # set WandB logging
        self.using_WandB = config['hyperparam_tuning']['WandB']['IsEnabled']
        if (self.using_WandB):
            # Log into weights and biases
            WandB_hpt.login(config)

        # set hyperparameter tuning
        self.hyperparam_tuning_mode = config['hyperparam_tuning']['optuna']['isEnabled']
        if self.hyperparam_tuning_mode:
            # Set optuna
            self.hyperHandler = HyperTuning_Handler(config)
            #self.Optuna_study = Optuna_hpt.Optuna_initialise_study(config) 
        else:
            # just do cross-validation
            pass



    def run_experiment(self, config):
        
        if(config['general']['testMode']):
            logging.info("WARNING: ------------ TEST MODE IS ACTIVE -------------")

        if self.hyperparam_tuning_mode:
            self.hyperHandler.run_optimize_experiment()

        else:
            # just do cross-validation
            results = K_fold_cross_validation(config)

