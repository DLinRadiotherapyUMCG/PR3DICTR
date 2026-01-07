import copy
import os

from src.experiments.experimentHandler import experimentHandler
from src.evaluation.validate_on_test_set import validate_models_on_test_set

def run_input_removal_experiment(config, experiment_name):
    """
    This function runs the input removal experiment. It will retrain the model on the dataset with one of the inputs (CT, dose, segmentations, clinical features) removed.
    Usage: It can be called from a main.py file, as long as you give it a config. It will then use the experimentHandler class to run the experiment.
    """

    assert config['hyperparam_tuning']['optuna']['isEnabled'] == False, "This experiment is not compatible with hyperparameter tuning. Please set 'isEnabled' to False in the config file."

    # set the experiment name
    config['general']['experiment_name'] = experiment_name

    # these are all of the image keys mentioned in the config file (i.e. a list of what we need to loop through)
    all_image_keys = config['data']['image_keys']
    
    for image_key in all_image_keys:
        # make a deep copy of the config (probably not necessary, but just to be sure)
        experiment_config = copy.deepcopy(config)

        # remove the image key from the config
        experiment_config['data']['image_keys'] = [key for key in all_image_keys if key != image_key]

        # set the trial number (i.e. the folder name for this input)
        experiment_config["general"]["trialNumber"] = "no_" + image_key

        # train the model (using 5 fold cross-validation)
        expHandler = experimentHandler(experiment_config)
        expHandler.run_experiment(experiment_config)

        #trial_dir = r"/scratch/s3719332/rt_pred_results/HP_TRP_HigherRes2_BEST/Trial_32_SGD1_GN_101/" # config['general']['resultsCurrentDirectory']
        trial_dir = os.path.join(experiment_config["paths"]["results"], experiment_config['general']['experiment_name'], experiment_config["general"]["trialNumber"])
        # # run the models on the test set
        validate_models_on_test_set(experiment_config, trial_dir)


    # if the clinical features are present, then also train a model without those (i.e. only the images, no clinical features)
    if len(config['columns']['clinical_features']) > 0:
        experiment_config = copy.deepcopy(config)
        experiment_config['columns']['clinical_features'] = []
        experiment_config["general"]["trialNumber"] = "no_clinical_features"
        expHandler = experimentHandler(experiment_config)
        expHandler.run_experiment(experiment_config)

        #trial_dir = r"/scratch/s3719332/rt_pred_results/HP_TRP_HigherRes2_BEST/Trial_32_SGD1_GN_101/" # config['general']['resultsCurrentDirectory']
        trial_dir = os.path.join(experiment_config["paths"]["results"], experiment_config['general']['experiment_name'], experiment_config["general"]["trialNumber"])
        # # run the models on the test set
        validate_models_on_test_set(experiment_config, trial_dir)


        





