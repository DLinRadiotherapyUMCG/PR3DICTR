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
    
    print("PyTorch CUDA version:", torch.version.cuda)
    print("Is CUDA available:", torch.cuda.is_available())
    

    """

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
    config['model']['model_name'] = 'TransRP'
    config['general']['experiment_name'] = "TRP_subsets"
    config['general']['trialNumber'] = "only_photons_w_spatial_dropout"

    config['model']['TransRP']['image_encoder'] = 'densenet'
    config['model']['linear_units'] = []
    config['model']['TransRP']['clinical_features_method'] = 'cls'

    config['model']['modulated_backbone'] = False
    

    
    #config['model']['TransRP']['clinical_features_method'] = 'cls'

    config['data']['source'] = "PRI2MA"
    config['data']['dataset_csv'] = "MT_dataset_photons_only.csv"
    config['columns']['clinical_features'] = [
                      'Geslacht', 'Leeftijd',
                      #"Concurrent_Chemo_or_Bioradiation", 
                      #"Protons",
                      #'P16_Negatief', 'P16_Niet_bepaald', 'P16_Positief', 
                      #'Roken_Ja, in verleden', 'Roken_Ja, nog steeds', 'Roken_Nee',

                      'Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg', 
                      'Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4', 
                      'Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg',
                      'Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg', 
                      'Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg', 
                     # 
                       
                    #   'WHO_0', 'WHO_1', 'WHO_2', 'WHO_3', 'WHO_4', 
                    #   'T0', 'T1', 'T2', 'T3', 'T4', 'Tis', 'Tx', 
                    #   'N0', 'N1', 'N2', 'N3', 
                    #   
                      ]
    
    
    

    config['training']['batch_size'] = 2
    config['data']['dataloader']['dataset_type'] = 'cache'
    config['data']['dataloader']['num_workers'] = 16
    config['training']['gradient_accumulation_steps'] = 1
    config['training']['optimizer']['name'] = "SGD"
    config['training']['optimizer']['learning_rate'] = 0.01
    config['training']['optimizer']['momentum'] = 0.8
    config['training']['optimizer']['weight_decay'] = 0

    config['data']['augmentation']['mixup']['isEnabled'] = False

    

    #config['data']['augmentation']['mixup']['isEnabled'] = False

    #config['general']['testMode'] = True
    #config['data']['n_patients_total'] = 200
    # # MAIN: DL running class with hyperparameter optimization

    from src.experiments.single_endpoint_models import run_single_toxicity_models_experiment
    from src.experiments.endpoint_combinations import run_toxicity_combinations_experiment


    # experiment_name = "ST_TRP_photons_only"
    # run_single_toxicity_models_experiment(config, experiment_name)

    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)

        

    # # # TEST ENSEMBLE CODE

    # from src.evaluation.validate_on_test_set import validate_models_on_test_set

    # trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber']) # r"C:\Users\S.P.M. de Vette\OneDrive - UMCG\Desktop\pred_RT_results\Xerostomia_M06/ResNet18" # config['general']['resultsCurrentDirectory']
    # # # # run the models on the test set
    # validate_models_on_test_set(config, trial_dir)

    """

        
