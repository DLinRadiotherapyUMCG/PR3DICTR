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

import src.hyper_opt.WandB_functions as WandB_functions


if __name__ == '__main__':

    # Setup
    log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = get_config('Daniel/Baoqiang_OS')

    # Disable randomness
    set_random_seed(config['general']['seed'])

    using_WandB = config['hyperparam_tuning']['WandB']['isEnabled']
        
    if (using_WandB):
        print("logging in..........")
        # Log into weights and biases
        WandB_functions.login(config)
        print("Weights and Biases is active")
    

    from src.uncertainty.deep_ensemble import train_deep_ensemble_models, evaluate_deep_ensemble_models
    from src.uncertainty.MC_dropout import train_MC_dropout_model, collect_bayesian_forward_passes

    # train_MC_dropout_model(config)
    # collect_bayesian_forward_passes(config)


    # config['general']['experiment_name'] = "Deep ensemble"

    # train_deep_ensemble_models(config)
    # evaluate_deep_ensemble_models(config)

    # config['model']['linear_units'] = [256]
    # config['model']['linear_units_endpoint'] = [32]



    
    
    # ["Geslacht", "Leeftijd", "Dysphagia_W01_Grade0_1", "Dysphagia_W01_Grade2", "Dysphagia_W01_Grade3_4"]
    config['general']['trialNumber'] = "Baoqiang_LRC_2year"  # Set the trial number for the experiment
    #config['columns']['clinical_features'] = []
    # config['general']['experiment_name'] = "Deep Ensemble"   
    # train_deep_ensemble_models(config)
    # evaluate_deep_ensemble_models(config)

    
    config['general']['experiment_name'] = "MC Dropout" 
    train_MC_dropout_model(config)
    collect_bayesian_forward_passes(config, UQ_method="MC_dropout") # UQ_method='TTA')
    

    # config['general']['trialNumber'] = "Xerostomia_2"  # Set the trial number for TTA
    # config['columns']['labels'] = ["Xerostomia_M06"]  # Set the label for TTA
    # config['columns']['clinical_features'] = ["Geslacht", "Leeftijd", "Xerostomia_W01_Helemaal_niet", "Xerostomia_W01_Een_beetje", "Xerostomia_W01_Nogal_Heel_erg"]


    # train_deep_ensemble_models(config)
    # evaluate_deep_ensemble_models(config)

    
    config['general']['experiment_name'] = "TTA" 
    train_MC_dropout_model(config)
    collect_bayesian_forward_passes(config, UQ_method='TTA')
    
    config['general']['experiment_name'] = "Deep Ensemble"   
    train_deep_ensemble_models(config)
    evaluate_deep_ensemble_models(config)

    

