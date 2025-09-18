import logging
import os

import wandb

# Set working directory to --> "Pred_RT"
import sys
#from pathlib import Path
#path_src = os.getcwd()
#sys.path.insert(1, path_src)

from src.config_presets.tools.get_config import get_config
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

from src.experiments.experimentHandler import experimentHandler

import numpy as np
import random
import torch
from monai.utils import set_determinism
import argparse


if __name__ == '__main__':

    # Setup
    #log_level, args = parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='INFO', nargs='?')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])   # NOTE: PASS THE TRIAL SEEDS IN AS AN ARGUMENT !! 
    args = parser.parse_args()

    # Load the toxicity
    log_level = args.log

    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = get_config('Trial32_Config')

    # Disable randomness
    set_random_seed(config['general']['seed'])

    from src.experiments.training_data_amounts import run_dataset_amounts_experiment

    experiment_name = "Trial32_Dataset_Amounts"
    seeds = args.seeds
    
    dataset_amounts=[100, 200, 300, 400, 500, 600]
    disable_data_augmentation = False

    
    run_dataset_amounts_experiment(config, experiment_name, trial_seeds=seeds, dataset_amounts=dataset_amounts, disable_data_augmentation=disable_data_augmentation)
    

    #K_fold_cross_validation(config)

    # # MAIN: DL running class with hyperparameter optimization
    #
    #expHandler = experimentHandler(config)
    #expHandler.run_experiment(config)
    #####hyperClass.Stop()


    # TEST ENSEMBLE CODE

    #from src.evaluation.validate_on_test_set import validate_models_on_test_set

    # trial_dir = r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Desktop\pred_RT_results\Xerostomia_M06/ResNet18" # config['general']['resultsCurrentDirectory']
    # # # # run the models on the test set
    # validate_models_on_test_set(config, trial_dir)

