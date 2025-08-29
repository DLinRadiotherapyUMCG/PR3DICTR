from src.dataset.get_transforms import get_transforms
from src.dataset.load_dataset import load_dataset, generate_K_fold_cross_validation_splits
from src.dataset.get_dataloader import make_dataloader   
from src.dataset.cumulative_sampling import generate_training_data_subsamples
from src.utils.data_equalizer import get_delimiter, label_equalizer


def set_dataset_config_for_uncertainty(config):
    """
    Set the dataset configuration for uncertainty training.
    This function modifies the original config to include settings from the uncertainty config.
    """
    # Update the dataset configuration with uncertainty settings
    config['data']['kFolds']['n_splits'] = config['uncertainty']['dataset_split']['n_splits']
    #config['data']['kFolds']['n_iterations'] = config['uncertainty']['dataset_split']['n_iterations']
    config['data']['kFolds']['validation_size'] = config['uncertainty']['dataset_split']['validation_size']
    config['data']['kFolds']['split_strategy'] = config['uncertainty']['dataset_split']['split_strategy']



def get_dataset_for_uncertainty_experiments(config, UQ_method = "MC_dropout"):
    # override the dataloader settings
    set_dataset_config_for_uncertainty(config)

    # Load the dataset and dataloader
    # load the data, and make K-fold splits
    df_train_val, df_test = load_dataset(config)
    k_fold_dataframes_list = generate_K_fold_cross_validation_splits(config, df_train_val)

    if config['general']['dataset_amounts_experiment']:
        k_fold_dataframes_list = generate_training_data_subsamples(config, k_fold_dataframes_list)
        df_val = k_fold_dataframes_list[0]['val']  # the validation set is always the same

        df_train_list = [split['train'] for split in k_fold_dataframes_list]
        
        # over/under sampling of the training set
        if(config['data']['equalizer']['isEnabled']):
            df_train_list = [label_equalizer(train_i_df, config) for train_i_df in df_train_list]
    else:
        dataset_split_dict = k_fold_dataframes_list[0]  # the first split is used for training and validation
        train_data, df_val = dataset_split_dict['train'], dataset_split_dict['val']

        df_train_list = [train_data]

    return df_train_list, df_val, df_test




