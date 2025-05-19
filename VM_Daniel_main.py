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
    config = get_config('Trial32_Config_VM')

    # Disable randomness
    set_random_seed(config['general']['seed'])
    
    

    # CLS experiment
    config['general']['experiment_name'] = "W12_models"
    config['general']['trialNumber'] = "TRP"

    
    config['model']['TransRP']['clinical_features_method'] = 'm2'

    config['data']['source'] = "PRI2MA"
    config['data']['dataset_csv'] = "MT_dataset_with_all_structs.csv"
    config['columns']['clinical_features'] = [
                      'Geslacht', 'Leeftijd',
                      'Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg', 
                      'Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 
                      'Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg',
                      'Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg', 
                      'Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg', 
                      #'Chemotherapy'
                      ]
    
    #config['columns']['labels'] = ['Aspiration_M06', 'Dysphagia_M06', 'Sticky_M06', 'Taste_M06', 'Xerostomia_M06']
    config['columns']['labels'] = ['Aspiration_W12', 'Dysphagia_W12', 'Sticky_W12', 'Taste_W12', 'Xerostomia_W12']

    # config['model']['TransRP']['image_encoder'] = 'convnext'
    # #
    # config['model']['convnext']['model_size'] = 'tiny'
    # config['model']['convnext']['kernel_size'] = 3
    # config['model']['convnext']['patch_size'] = 5

    config['training']['batch_size'] = 4
    #config['training']['max_epochs'] = 3

    #config['data']['kFolds']['n_iterations'] = 1
    #config['data']['dataloader']['dataset_type'] = 'smartcache'

    #config['data']['augmentation']['mixup']['isEnabled'] = False

    #config['general']['testMode'] = True
    #config['data']['n_patients_total'] = 200
    # # MAIN: DL running class with hyperparameter optimization
    #
    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)



    # TEST ENSEMBLE CODE

    from src.evaluation.validate_on_test_set import validate_models_on_test_set

    trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber']) # r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Desktop\pred_RT_results\Xerostomia_M06/ResNet18" # config['general']['resultsCurrentDirectory']
    # # # run the models on the test set
    validate_models_on_test_set(config, trial_dir)

