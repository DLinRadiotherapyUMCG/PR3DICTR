import numpy as np
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


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
        #rint(f"Sampled {len(df_train_sample)} patients")

    logging.info(len(k_fold_dataframes_list), len(config['data']['n_training_patients_list']))
    return k_fold_dataframes_list
