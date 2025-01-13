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
    config = get_config('HP_TRP_HigherRes2_BEST')

    # Disable randomness
    set_random_seed(config['general']['seed'])
    

    #K_fold_cross_validation(config)

    endpoints_list = ["Aspiration_M06", "Dysphagia_M06", "Sticky_M06", "Taste_M06", "Xerostomia_M06" ]

    general_features = ['Sex', 'Age']
    baseline_features_dict = {
        'Aspiration_M06' : ['Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg'],
        'Dysphagia_M06' : ['Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4'],
        'Sticky_M06' : ['Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg'],
        'Taste_M06' : ['Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg'],
        'Xerostomia_M06' : ['Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg'], 
    }

    for endpoint in endpoints_list:
        config = get_config('HP_TRP_HigherRes2_BEST')

        # Disable randomness
        set_random_seed(config['general']['seed'])

        config['general']['experiment_name'] = "TRP_HigherRes2_BEST_ST_models"
        config['general']['trialNumber'] = endpoint

        config['columns']['labels'] = [endpoint]
        config['columns']['clinical_features'] = general_features + baseline_features_dict[endpoint]

        config['data']['stratify_on'] = [endpoint]
        config['training']['loss']['BCE']['pos_weight'] = [1]

        # # MAIN: DL running class with hyperparameter optimization
        #
        expHandler = experimentHandler(config)
        expHandler.run_experiment(config)
        #####hyperClass.Stop()

        # # MAIN: DL running class with hyperparameter optimization
        #
        #expHandler = experimentHandler(config)
        #expHandler.run_experiment(config)
        #####hyperClass.Stop()


        # TEST ENSEMBLE CODE

        from src.evaluation.validate_on_test_set import validate_models_on_test_set


        #trial_dir = r"/scratch/s3719332/rt_pred_results/HP_TRP_HigherRes2_BEST/Trial_32_SGD1_GN_101/" # config['general']['resultsCurrentDirectory']
        
        trial_dir = r"/scratch/s3719332/rt_pred_results/TRP_HigherRes2_BEST_ST_models/TRP_HigherRes2_BEST_ST_models/" # config['general']['resultsCurrentDirectory']
        trial_dir = os.path.join(trial_dir, endpoint)
        # # run the models on the test set
        validate_models_on_test_set(config, trial_dir)
    
    