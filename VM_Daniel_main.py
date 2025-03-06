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

    print(torch.compiler.list_backends())

    # Setup
    log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = get_config('Trial32_Config_VM')

    # Disable randomness
    set_random_seed(config['general']['seed'])
    
    
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




    # CLS experiment
    config['general']['experiment_name'] = "Trial32_CLS_method_testing_2"
    config['general']['trialNumber'] = "CLS_no_gating_output_mixing"

    config['model']['TransRP']['clinical_features_method'] = 'cls'
    config['model']['TransRP']['cls_hidden_dim'] = 64
    config['model']['TransRP']['cls_gating'] = False

    config["model"]["TransRP"]["cls_merge_image_features"] = False
    config["model"]["TransRP"]["cls_per_class_weighting"] = True
    

    # # MAIN: DL running class with hyperparameter optimization
    #
    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)



    # TEST ENSEMBLE CODE

    from src.evaluation.validate_on_test_set import validate_models_on_test_set

    trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber']) # r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Desktop\pred_RT_results\Xerostomia_M06/ResNet18" # config['general']['resultsCurrentDirectory']
    # # # run the models on the test set
    validate_models_on_test_set(config, trial_dir)

