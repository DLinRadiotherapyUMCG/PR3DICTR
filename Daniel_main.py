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

if __name__ == '__main__':

    # Setup
    log_level = parse_args()
    setup_logging(log_level)

    # Load the config
    config = get_config('Daniel/SPARK.yaml')

    # Disable randomness
    set_random_seed(config['general']['seed'])
    

    # # MAIN: DL running class (with optional hyperparameter optimization)
    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)


    # TEST ENSEMBLE CODE
    from src.evaluation.validate_on_test_set import validate_models_on_test_set

    # # # # run the models on the test set
    # trial_dir = config['general']['resultsCurrentDirectory']
    # trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
    # validate_models_on_test_set(config, trial_dir)

