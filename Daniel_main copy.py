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
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    

    methods = ["cls", "m2", "m2_v2", "m2_cls"]
    #methods = ['cls']

    for method in methods:
        config = get_config('Multi_tox')

        config['model']['TransRP']['clinical_features_method'] = method
        config['paths']['results'] = rf"C:/Users/S.P.M. de Vette/OneDrive - UMCG/Desktop/pred_RT_results"
        config['general']['trialNumber'] = method
        # Disable randomness
        set_random_seed(config['general']['seed'])
        expHandler = experimentHandler(config)
        expHandler.run_experiment(config)


        # TEST ENSEMBLE CODE
        from src.evaluation.validate_on_test_set import validate_models_on_test_set

        trial_dir = rf"C:/Users/S.P.M. de Vette/OneDrive - UMCG/Desktop/pred_RT_results/CLC_method2/{method}"

        #trial_dir = rf"C:/Users/S.P.M. de Vette/OneDrive - UMCG/Desktop/pred_RT_results/CLC_methods/{method}" # " # config['general']['resultsCurrentDirectory']
        # # run the models on the test set
        validate_models_on_test_set(config, trial_dir)

