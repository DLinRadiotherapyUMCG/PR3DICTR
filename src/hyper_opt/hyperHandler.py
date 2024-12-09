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



class HyperTuning_Handler():
    def __init__(self,config):
        # Set optuna
        self.Optuna_study = Optuna_hpt.Optuna_initialise_study(config) 
        self.config = config



        # NOTE: deprecated. is now in experimentHandler
        # self.using_WandB = config['hyperparam_tuning']['WandB']['IsEnabled']
        # if (self.using_WandB):
        #     # Log into weights and biases
        #     WandB_hpt.login(config)


    def run_optimize_experiment(self):
        """
        Run the hyperparameter tuning experiment
        """
        
        # if(self.config['general']['testMode']):
        #     logging.info("WARNING: ------------ TEST MODE IS ACTIVE -------------")

        # self.Optuna_study.optimize(
        #     func = run_trial(self, config, trial)
        # )
        n_trials = self.config['hyperparam_tuning']['optuna']['n_trials']

        self.Optuna_study.optimize(lambda trial: self.run_trial(self.config, trial), n_trials)



    def run_trial(self, config, trial):
        """
        Manages a single trial of the hyperparameter tuning experiment. 
        Updates the config with the new hyperparameters, runs the K-fold cross validation and reports the results to Optuna.
        Args:
            config (dict): the configuration of the experiment
            trial (optuna.trial.Trial): the trial object of the current trial
        Returns:
            list: the results of the trial
        """
        # set a new trial number (also used as the 'group' for WandB later)
        config['general']['trialNumber'] = f"Trial_{trial.number}" 

        # set random seed
        set_random_seed(config['general']['seed'])

        # update the config of this trial
        config = self.update_trial_hyperparameters(config, trial)

        # K-fold cross validation here
        trial_hyper_param_dict = trial.params
        all_results = K_fold_cross_validation(config, config_for_wandb=trial_hyper_param_dict)

        # aggregate the results of the K-folds, report them to Optuna
        optuna_results = self.results_handler(config, all_results)

        return optuna_results
        # end.
        # 

    def update_trial_hyperparameters(self, config, trial):
        """
        Updates the config with the new hyperparameters for the current trial
        Args:
            config (dict): the configuration of the experiment
            trial (optuna.trial.Trial): the trial object of the current trial
        Returns:
            updated_config (dict): the updated configuration, with trial parameters overwriting the default ones
        """

        # Alter normal hyperparameters
        updated_config = Optuna_hpt.normal_hyperparameters(config, trial)

        # Alter any hyperparameters that are dependend on other parameters 
        updated_config = Optuna_hpt.derived_hyperparameters(updated_config, trial)

        return updated_config


    def results_handler(self, config, results):
        """
        reports the desired metric/loss values to optimise to Optuna
        The K-fold cross validation results contain a lot of info, this function extracts the ones we want to optimise.
        Args:
            config (dict): the configuration of the experiment
            results (dict): the results of the K-fold cross validation
        Returns:
            trial_result (list): the results of the trial
        """

        objectives = config['hyperparam_tuning']['optuna']['objectives'] 

        results_keys = list(results.keys())

        for metric in objectives:
            if metric not in results_keys:
                raise Exception(f"Metric {metric} is not in the results dictionary")
            

        trial_result = [results[metric] for metric in objectives]

        return trial_result




    

