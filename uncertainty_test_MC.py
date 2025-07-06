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
    config = get_config('Daniel/VM_uncertainty')

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

    config['general']['experiment_name'] = "MC_Dropout"


    dropout_values = ['05', '10', '15', '20', '25', '30', '35', '40', '45', '50']  # String format for WandB


    # first for DYSPHAGIA

    config['columns']['labels'] = ["Dysphagia_M06"]  # Set the label for TTA
    config['columns']['clinical_features'] = ["Dysphagia_W01_Grade0_1", "Dysphagia_W01_Grade2", "Dysphagia_W01_Grade3_4",
                                              "Xerostomia_W01_Helemaal_niet", "Xerostomia_W01_Een_beetje", "Xerostomia_W01_Nogal_Heel_erg",
                                                'Loctum2_Larynx', 'Loctum2_Oral_Cavity',  'Loctum2_Pharynx','Loctum2_Overig',
                                                'P16_Negatief', 'P16_Niet_bepaald', 'P16_Positief', 
                                                'WHO_0', 'WHO_1', 'WHO_2', 'WHO_3', 'WHO_4', 
                                                'T0', 'T1', 'T2', 'T3', 'T4', 'Tis', 'Tx', 
                                                'N0', 'N1', 'N2', 'N3', 
                                                'Roken_Ja, in verleden', 'Roken_Ja, nog steeds', 'Roken_Nee'
                                                ]

    for dropout_value in dropout_values:
        config['general']['trialNumber'] = "Dysphagia_" + dropout_value
        config['model']['dropout_p'] = float(dropout_value) / 100.0  # Convert to float for the model

        set_random_seed(config['general']['seed'])

        train_MC_dropout_model(config)
        collect_bayesian_forward_passes(config, UQ_method='MC_dropout')

    # then for XEROSTOMIA

    config['columns']['labels'] = ["Xerostomia_M06"]  # Set the label for TTA
    config['columns']['clinical_features'] = ["Geslacht", "Leeftijd", "Xerostomia_W01_Helemaal_niet", "Xerostomia_W01_Een_beetje", "Xerostomia_W01_Nogal_Heel_erg"]
    
    for dropout_value in dropout_values:
        config['general']['trialNumber'] = "Xerostomia_" + dropout_value
        config['model']['dropout_p'] = float(dropout_value) / 100.0

        set_random_seed(config['general']['seed'])

        train_MC_dropout_model(config)
        collect_bayesian_forward_passes(config, UQ_method='MC_dropout')

