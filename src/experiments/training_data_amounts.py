from src.experiments.experimentHandler import experimentHandler

import numpy as np
import random
import torch
from monai.utils import set_determinism
import copy
import os
from src.utils.set_random_seed import set_random_seed, generate_random_seed






def run_dataset_amounts_experiment(config, experiment_name, n_trials, dataset_amounts):
    """
    This function runs the an experiment using different combinations of endpoints removal experiment. It will retrain the model on the dataset with one of the inputs (CT, dose, segmentations, clinical features) removed.
    Usage: It can be called from a main.py file, as long as you give it a config. It will then use the experimentHandler class to run the experiment.
    """

    assert config['hyperparam_tuning']['optuna']['isEnabled'] == False, "This experiment is not compatible with hyperparameter tuning. Please set 'isEnabled' to False in the config file."

    # set the experiment name
    config['general']['experiment_name'] = experiment_name

    # we are doing this experiment by bootstrapping, so we need to set this to True
    config['general']['bootstrapping_experiment'] = True
    # and force the N iterations to be 1
    config['data']['kFolds']['n_iterations'] = 1
    config['general']['use_test_set'] = True



    for trial_idx in range(n_trials):
        
        experiment_config = copy.deepcopy(config)
        experiment_config['general']['seed'] = generate_random_seed()  # set a new random seed
        
        for dataset_amount in dataset_amounts:
            set_random_seed(experiment_config['general']['seed'])           # make sure that the seed is set for each dataset amount !!!
            

            experiment_config['general']['trialNumber'] = f"{dataset_amount}_training_patients"

            experiment_config['data']['subsample_train_set_size'] = dataset_amount  # set in the config how many training patients we want to use

            # train model for 1 fold
            expHandler = experimentHandler(experiment_config)
            expHandler.run_experiment(experiment_config)


        """
        # TEST ENSEMBLE CODE

        #trial_dir = r"/scratch/s3719332/rt_pred_results/HP_TRP_HigherRes2_BEST/Trial_32_SGD1_GN_101/" # config['general']['resultsCurrentDirectory']
        trial_dir = os.path.join(experiment_config["paths"]["results"], experiment_config['general']['experiment_name'], experiment_config['general']['trialNumber'])
        # # run the models on the test set
        validate_models_on_test_set(experiment_config, trial_dir)
        """
