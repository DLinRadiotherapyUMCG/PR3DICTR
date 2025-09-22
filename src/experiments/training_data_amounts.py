import copy

from src.utils.set_random_seed import set_random_seed, generate_random_seed
from src.experiments.experimentHandler import experimentHandler



def run_dataset_amounts_experiment(config, experiment_name, trial_seeds, dataset_amounts, disable_data_augmentation=True):
    """
    
    """

    assert config['hyperparam_tuning']['optuna']['isEnabled'] == False, "This experiment is not compatible with hyperparameter tuning. Please set 'isEnabled' to False in the config file."

    # set the experiment name
    config['general']['experiment_name'] = experiment_name

    # we are doing this experiment by bootstrapping, so we need to set this to True
    config['general']['dataset_amounts_experiment'] = True
    config['data']['n_training_patients_list'] = dataset_amounts

    # and force the N iterations to be 1
    config['data']['kFolds']['n_iterations'] = 1
    config['general']['use_test_set'] = True

    if disable_data_augmentation:
        config['data']['augmentation']['isEnabled'] = False
        config['data']['augmentation']['mixup']['isEnabled'] = False


    for trial_idx, trial_seed in enumerate(trial_seeds):
        
        experiment_config = copy.deepcopy(config)
        experiment_config['general']['seed'] = trial_seed #generate_random_seed()  # set a new random seed
        
        #for dataset_amount in dataset_amounts:
        set_random_seed(experiment_config['general']['seed'])           # make sure that the seed is set for each dataset amount !!!

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



