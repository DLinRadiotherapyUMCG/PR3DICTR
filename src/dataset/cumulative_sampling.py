import numpy as np
import logging
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def perform_cumulative_sampling(config, df: pd.DataFrame, sampled_indices: list, target_sample_size: int) -> tuple[pd.DataFrame, list]:
    """
    Perform cumulative sampling to select a target number of samples from the dataframe.
    This function ensures that the sampled indices are cumulative, meaning that each time it is called,
    it adds new samples to the existing list of sampled indices without removing any previously sampled ones.
    Args:
        config (dict): configuration dictionary
        df (pd.DataFrame): dataframe to sample from
        sampled_indices (list): list of already sampled indices
        target_sample_size (int): target number of samples to select
    Returns:
        df_sampled (pd.DataFrame): dataframe containing the sampled rows
        sampled_indices (list): updated list of sampled indices
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
        # shuffle the indicies of the remaining patients and select the required number of samples
        shuffle_split = ShuffleSplit(n_splits=1, train_size=n_required_samples) # , random_state=config['general']['seed'])
        new_indices, _ = next(shuffle_split.split(remaining_indices))
        new_indices = np.array(remaining_indices)[new_indices]  # Convert back to the original index space

    # update the list of sampled indices with the new_indices
    sampled_indices += list(new_indices)

    # check that the number of unique indices is equal to the target sample size
    assert len(sampled_indices) == len(set(sampled_indices)) == target_sample_size
    
    return df.iloc[sampled_indices], sampled_indices



def generate_training_data_subsamples(config, k_fold_dataframes_list: list) -> list:
    """
    Generate cumulative training data subsamples for each fold in the k-fold cross-validation splits.
    This function uses the first fold of the K-folds dataframes list to create cumulative training data subsamples (i.e., increasing amounts of training data).
    The validation set remains the same for all subsamples.
    The function returns a new list of k-fold dataframes, where each entry corresponds to a different amount of training data.
    Args:
        config (dict): configuration dictionary
        k_fold_dataframes_list (list): list of dictionaries containing 'train' and 'val' dataframes for each fold
    Returns:
        k_fold_dataframes_list: list of dictionaries containing 'train' and 'val' dataframes for each subsample
    """
    dataset_split_dict = k_fold_dataframes_list[0]
    df_all_train_patients, df_val = dataset_split_dict['train'], dataset_split_dict['val']

    k_fold_dataframes_list = []
    sampled_indices = []
        
    for n_training_patients in config['data']['n_training_patients_list']:
        df_train_sample, sampled_indices = perform_cumulative_sampling(config, df_all_train_patients, sampled_indices, n_training_patients)

        k_fold_dataframes_list.append({'train': df_train_sample, 'val': df_val})
        logging.info(f"Sampled {len(df_train_sample)} patients")

    return k_fold_dataframes_list
