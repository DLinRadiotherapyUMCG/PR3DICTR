import logging
import os

import wandb

# Set working directory to --> "Pred_RT"
import sys
#from pathlib import Path
#path_src = os.getcwd()
#sys.path.insert(1, path_src)

from src.evaluation.validate_on_test_set import validate_models_on_test_set
from src.config_presets.tools.get_config import get_config
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

from src.experiments.experimentHandler import experimentHandler

import numpy as np
import random
import torch
from monai.utils import set_determinism

CUDA_LAUNCH_BLOCKING=1



def normal_model(config):
    config['training']['loss']['BCE']['pos_weight'] = [1]
    config['data']['equalizer']['isEnabled'] = False

    config['general']['trialNumber'] = "baseline"

    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)

    trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
    validate_models_on_test_set(config, trial_dir)


def pos_weight_model(config):
    config['training']['loss']['BCE']['pos_weight'] = 'auto'
    config['data']['equalizer']['isEnabled'] = False

    config['general']['trialNumber'] = "BCE pos_weight"

    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)

    trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
    validate_models_on_test_set(config, trial_dir)


def oversampling_model(config):
    config['training']['loss']['BCE']['pos_weight'] = [1]
    config['data']['equalizer']['isEnabled'] = True
    config['data']['equalizer']['resamplingDirection'] = 'over'

    config['general']['trialNumber'] = "BCE oversampling"

    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)

    trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
    validate_models_on_test_set(config, trial_dir)

    
def undersampling_model(config):
    config['training']['loss']['BCE']['pos_weight'] = [1]
    config['data']['equalizer']['isEnabled'] = True
    config['data']['equalizer']['resamplingDirection'] = 'under'

    config['general']['trialNumber'] = "BCE undersampling"

    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)

    trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
    validate_models_on_test_set(config, trial_dir)


if __name__ == '__main__':

    # Clear torch cache
    import torch
    torch.cuda.init()
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Setup
    log_level = parse_args()
    setup_logging(log_level)

    # Load the config
    config = get_config('Daniel/ORN/DL_ORN')

    # Disable randomness
    set_random_seed(config['general']['seed'])
    config['general']['experiment_name'] = "Mandible ORN_4 model"
    config['columns']['labels'] = ['ORN_4']
    config['data']['stratify_on'] = ['ORN_4']

    # normal_model(config)
    # pos_weight_model(config)
    # oversampling_model(config)
    undersampling_model(config)
    

    # # MAIN: DL running class (with optional hyperparameter optimization)
    # expHandler = experimentHandler(config)
    # expHandler.run_experiment(config)


    # TEST ENSEMBLE CODE
    

    # # # # # run the models on the test set
    # trial_dir = config['general']['resultsCurrentDirectory']
    # trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
    # validate_models_on_test_set(config, trial_dir)

