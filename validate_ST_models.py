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


from src.evaluation.validate_on_test_set import validate_models_on_test_set

def old_main():

    # Setup
    log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='t
  
    #config_dir = f"/scratch/p306883/rt_pred_results/HP_newData_1/Trial_{trial_n}/KFold1/DlModel_Config"
    
    config = get_config('DlModel_Config', pathGiven = f"/scratch/p306883/rt_pred_results/Feb_2025/Trial_32/KFold1")

    # Disable randomness
    set_random_seed(config['general']['seed'])
    
    
    config['data']['source'] = 'PRIMA'
    config['data']['dataset_csv'] = "MT_post_2021_validation_set.csv"
    config['paths']['csv'] = "/scratch/p306883/PRIMA_dataset"
    config['paths']['images'] = "/scratch/p306883/PRIMA_dataset/dataset_full_renamed"
    

    
  
    #trial_dir = r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Desktop\pred_RT_results\Xerostomia_M06/ResNet18" # config['general']['resultsCurrentDirectory']
    trial_dir = os.path.join('/scratch/p306883/rt_pred_results/Feb_2025//Trial_32/')
    # # # # run the models on the test set
    validate_models_on_test_set(config, trial_dir)
    
    

if __name__ == '__main__':
    
    endpoint_list = ['Aspiration_M06', 'Dysphagia_M06', 'Sticky_M06', 'Taste_M06', 'Xerostomia_M06']
    ST_models_folder_dir = r'\\zkh\appdata\RTDicom\Projectline_HNC_modelling\Users\Daniel MacRae\1. MultiTox_HNC\Feb_2025_results\Trial32_singletox_models'
    
    for endpoint in endpoint_list:
        trial_dir = os.path.join(ST_models_folder_dir, endpoint)
        
        config = get_config('DlModel_Config', pathGiven = os.path.join(trial_dir, "KFold1"))
        
        set_random_seed(config['general']['seed'])
        
        config['data']['source'] = 'PRIMA'
        config['data']['dataset_csv'] = "MT_post_2021_validation_set.csv"
        config['paths']['csv'] = r"\\zkh\appdata\RTDicom\Projectline_HNC_modelling\Users\Daniel MacRae\1. MultiTox_HNC\Dataset"
        config['paths']['images'] = r"\\zkh\appdata\RTDicom\Projectline_HNC_modelling\PRI2MA\all_data_preprocessed_2025\dataset_full_anonymized"
        
        #trial_dir = os.path.join('/scratch/p306883/rt_pred_results/Feb_2025//Trial_32/')
        # # # # run the models on the test set
        validate_models_on_test_set(config, trial_dir)
    
      
      

