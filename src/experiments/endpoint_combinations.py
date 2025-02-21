from src.experiments.experimentHandler import experimentHandler

import numpy as np
import random
import torch
from monai.utils import set_determinism
import copy
import os

from src.utils.set_random_seed import set_random_seed
from src.evaluation.validate_on_test_set import validate_models_on_test_set


def get_single_tox_feature_set(endpoint):

    endpoint_features_dict = {"Aspiration_M06" : ['Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg'],
                              "Dysphagia_M06" : ['Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4'],
                              "Sticky_M06" : ['Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg'],
                              "Taste_M06" : ['Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg'],
                              "Xerostomia_M06" : ['Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg'],
                              }
    
    return endpoint_features_dict[endpoint]





def run_toxicity_combinations_experiment(config, experiment_name, endpoint_combinations_dict=None):
    """
    This function runs the an experiment using different combinations of endpoints removal experiment. It will retrain the model on the dataset with one of the inputs (CT, dose, segmentations, clinical features) removed.
    Usage: It can be called from a main.py file, as long as you give it a config. It will then use the experimentHandler class to run the experiment.
    """

    assert config['hyperparam_tuning']['optuna']['isEnabled'] == False, "This experiment is not compatible with hyperparameter tuning. Please set 'isEnabled' to False in the config file."
    
    # set the experiment name
    config['general']['experiment_name'] = experiment_name
    
    # these are all of the image keys mentioned in the config file (i.e. a list of what we need to loop through)
    original_endpoints = config['columns']['labels']

    if endpoint_combinations_dict == None:
        endpoint_combinations_dict = {
            "salivary_domain" : ["Sticky_M06", "Taste_M06", "Xerostomia_M06"],
            "no taste" : ["Aspiration_M06", "Dysphagia_M06", "Sticky_M06", "Xerostomia_M06"],
            "swallowing_domain" : ["Aspiration_M06", "Dysphagia_M06"],
        }

    main_features = ['Sex', 'Age', # "Chemotherapy",
                #'Loctum2_Larynx', 'Loctum2_Oral_Cavity', 'Loctum2_Overig', 'Loctum2_Pharynx',
                ]

    for key, endpoints in endpoint_combinations_dict.items():
        # make a deep copy of the config (probably not necessary, but just to be sure)
        experiment_config = copy.deepcopy(config)
        set_random_seed(experiment_config['general']['seed'])
        

        experiment_config['columns']['labels'] = endpoints
        config['data']['stratify_on'] = endpoints

        # include the clinical features for these toxicities (i.e. not the baselines for toxicities which are not included)
        toxicity_clinical_features_set = main_features
        for endpoint in endpoints:
            toxicity_clinical_features_set += get_single_tox_feature_set(endpoint)

        experiment_config['columns']['clinical_features'] = toxicity_clinical_features_set

        # set the trial number (i.e. the folder name for this input)
        experiment_config["general"]["trialNumber"] = key

        # set the weights of BCE loss (otherwise it will crash because the number of classes is different)
        config['training']['loss']['BCE']['pos_weight'] = [1.0] * len(endpoints)

        # train the model (using 5 fold cross-validation)
        expHandler = experimentHandler(experiment_config)
        expHandler.run_experiment(experiment_config)

        # TEST ENSEMBLE CODE

        #trial_dir = r"/scratch/s3719332/rt_pred_results/HP_TRP_HigherRes2_BEST/Trial_32_SGD1_GN_101/" # config['general']['resultsCurrentDirectory']
        trial_dir = os.path.join(config["paths"]["results"], config['general']['experiment_name'], config['general']['trialNumber'])
        # # run the models on the test set
        validate_models_on_test_set(config, trial_dir)
