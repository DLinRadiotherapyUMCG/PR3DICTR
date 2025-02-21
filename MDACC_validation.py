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
from src.evaluation.validate_on_test_set import validate_models_on_test_set

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

    config = get_config('Transrp_daniel_check_Taste_M06')

    # Disable randomness
    set_random_seed(config['general']['seed'])
    
    

    """
    OVERWRITE THINGS IN THE CONFIG TO FORCE IT TO LOOK AT MDACC
    """

    config['data']['source'] = 'MDACC'
    config['data']['dataset_csv'] = "MDACC_stratified_sampling_test.csv"

    config['paths']['csv'] = r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Documents\DL_NTCP_Multitox\datasets\MDACC_dataset"
    config['paths']['images'] = r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Documents\DL_NTCP_Multitox\datasets\MDACC_dataset\mdacc_patients_high_res"
    #config['data']['paths']['results'] = r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Desktop\pred_RT_results\Xerostomia_M06\MDACC\masks"

    
    # directory to folds folder
    trial_dir = r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Desktop\PRIMA taste\results\Attentionmaps\Transrp_highres\TRP_Taste" # config['general']['resultsCurrentDirectory']
    #config['general']['resultsCurrentDirectory'] = trial_dir
    # # # # run the models on the test set
    validate_models_on_test_set(config, trial_dir)

