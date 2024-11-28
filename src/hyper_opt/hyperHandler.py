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

from torch.utils.data import DataLoader
import os
import numpy as np

from functools import partial

import src.hyper_opt.Optuna_hpt as Optuna_hpt
import src.hyper_opt.WandB_hpt as WandB_hpt



class HyperTuning_Handler():
    def __init__(self,config):
        # Set optuna
        self.Optuna_study = Optuna_initialise_study(config) 
        self.config = config


        self.using_WandB = config['hyperparam_tuning']['WandB']['IsEnabled']

        if (self.using_WandB):
            # Log into weights and biases
            WandB_hpt.login(config)


    def run_experiment(self, config):
        
        if(config['general']['testMode']):
            logging.info("WARNING: ------------ TEST MODE IS ACTIVE -------------")

        # self.Optuna_study.optimize(
        #     func = run_trial(self, config, trial)
        # )
        n_trials = config['hyperparam_tuning']['optuna']['n_trials']

        self.Optuna_study.optimize(lambda trial: run_trial(config, trial), n_trials)



def results_handler(config, results):
    """
    reports the desired metric/loss values to optimise to Optuna
    The K-fold cross validation results contain a lot of info, this function extracts the ones we want to optimise
    """

    objectives = config['hyperparam_tuning']['optuna']['objectives'] 

    results_keys = list(results.keys())

    for metric in objectives:
        if metric not in results_keys:
            raise Exception(f"Metric {metric} is not in the results dictionary")
        

    trial_result = [results[metric] for metric in objectives]

    return trial_result




def Optuna_initialise_study(config):
    study = None
    if (config['hyperparam_tuning']['optuna']['IsEnabled']):

        # Check where to keep study information
        studyName = config['general']['experiment_name']
        pathOptunaStudyTracker = os.path.join(os.path.join(config['paths']['results'] , studyName), "track_optuna.db")
        storage_name = f"sqlite:///{pathOptunaStudyTracker}"
        create_file(pathOptunaStudyTracker)
        print(storage_name)

        study = optuna.create_study(
                            study_name = studyName, 
                            storage = storage_name, 
                            load_if_exists=True,
                            directions= config['hyperparam_tuning']['optuna']['objective_direction']
                            )

        # check if an experiment has been run before
        config['general']['firstRun'] = (len(study.get_trials()) == 0)
        if (config['general']['firstRun']):
            print("First optuna run has been detected!")

    return study




def run_trial(config, trial):
    # set a new trial number (also used as the 'group' for WandB later)
    config['general']['trialNumber'] = f"Trial_{trial.number}" 

    # update the config of this trial
    config = update_trial_hyperparameters(config, trial)


    # K-fold cross validation here
    all_results = K_fold_cross_validation(config)

    # aggregate the results of the K-folds, report them to Optuna
    optuna_results = results_handler(config, all_results)

    return optuna_results
    # end.
    
    


def update_trial_hyperparameters(config, trial):
    # Alter normal hyperparameters
    config = Optuna_hpt.normal_hyperparameters(config, trial)

    # Alter any hyperparameters that are dependend on other parameters 
    config = Optuna_hpt.derived_hyperparameters(config, trial)

    return config




def update_config(dic, location, suggested_value):
    if len(location) == 1:
        dic[location[0]] = suggested_value
        return dic
    else:
        dic[location[0]] = update_config(dic[location[0]], location[1:], suggested_value)
        return dic
    

    
    
    