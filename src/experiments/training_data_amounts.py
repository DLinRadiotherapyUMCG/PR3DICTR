import numpy as np
import random
import torch
from monai.utils import set_determinism
import copy
import os

from src.utils.set_random_seed import set_random_seed, generate_random_seed
from src.experiments.experimentHandler import experimentHandler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit



def run_dataset_amounts_experiment(config, experiment_name, trial_seeds, dataset_amounts, disable_data_augmentation=True):
    """
    This function runs the an experiment using different combinations of endpoints removal experiment. It will retrain the model on the dataset with one of the inputs (CT, dose, segmentations, clinical features) removed.
    Usage: It can be called from a main.py file, as long as you give it a config. It will then use the experimentHandler class to run the experiment.
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

        #experiment_config['general']['trialNumber'] = experiment_config['general']['seed'] 
            

            #experiment_config['general']['trialNumber'] = f"{dataset_amount}_training_patients"

            #experiment_config['data']['subsample_train_set_size'] = dataset_amount  # set in the config how many training patients we want to use

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













def perform_cumulative_sampling(config, df, sampled_indices, target_sample_size):
    """
    """
    
    remaining_indices = list(set(range(len(df))) - set(sampled_indices))  # determine which patients have not already been sampled
    n_required_samples = target_sample_size - len(sampled_indices)  # how many more patients  we need to sample
    
    if n_required_samples == len(remaining_indices):
        new_indices = remaining_indices
    elif n_required_samples < 0:
        # if the number of required samples is less than 0, raise an error
        # i.e. there is not enough training data to sample the target number of patients
        raise ValueError(f"Number of required samples is less than 0. There is not enough training data to sample {target_sample_size} patients.")
    else:
        # NOTE: it is not always possible to do stratified sampling. Often there is just one patient with a certain combination of feature values, which makes splitting impossible
        # NOTE: so, we have only implemented random splitting for this experiment
        # if config["data"]["kFolds"]["split_strategy"] == 'stratified':
        #     labels = df[config['data']['stratify_on']]
        #     labels_string = [''.join(str(l)) for l in labels.values]
        #     labels_string = [''.join(str(l).replace(' ', '').replace('-', '')) for l in labels.values]
        #     encoded_labels = LabelEncoder().fit_transform(labels_string)
        #     shuffle_split = StratifiedShuffleSplit(n_splits=1, train_size=n_required_samples)
        #     new_indices, _ = next(shuffle_split.split(df.iloc[remaining_indices], encoded_labels[remaining_indices]))
        #     new_indices = np.array(remaining_indices)[new_indices]

            # strat_split = StratifiedShuffleSplit(n_splits=1, train_size=required_samples, random_state=random_state)
            # new_indices, _ = next(strat_split.split(X[remaining_indices], y[remaining_indices]))
            # new_indices = np.array(remaining_indices)[new_indices]

        #else:

        # shuffle the indicies of the remaining patients and select the required number of samples
        shuffle_split = ShuffleSplit(n_splits=1, train_size=n_required_samples) # , random_state=config['general']['seed'])
        new_indices, _ = next(shuffle_split.split(remaining_indices))
        new_indices = np.array(remaining_indices)[new_indices]  # Convert back to the original index space

    # update the list of sampled indices with the new_indices
    sampled_indices += list(new_indices)

    # check that the number of unique indices is equal to the target sample size
    assert len(sampled_indices) == len(set(sampled_indices)) == target_sample_size
    
    return df.iloc[sampled_indices], sampled_indices



def generate_training_data_subsamples(config, k_fold_dataframes_list):
    dataset_split_dict = k_fold_dataframes_list[0]
    df_all_train_patients, df_val = dataset_split_dict['train'], dataset_split_dict['val']

    k_fold_dataframes_list = []
    sampled_indices = []
        
    for n_training_patients in config['data']['n_training_patients_list']:
            #config["data"]['n_training_patients'] = n_training_patients
        df_train_sample, sampled_indices = perform_cumulative_sampling(config, df_all_train_patients, sampled_indices, n_training_patients)

        k_fold_dataframes_list.append({'train': df_train_sample, 'val': df_val})
        print(f"Sampled {len(df_train_sample)} patients")

    print(len(k_fold_dataframes_list), len(config['data']['n_training_patients_list']))
    return k_fold_dataframes_list
