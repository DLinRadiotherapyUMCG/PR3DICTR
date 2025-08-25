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
    train_transforms, val_transforms = get_transforms(config)


    if config['general']['dataset_amounts_experiment']:
        k_fold_dataframes_list = generate_training_data_subsamples(config, k_fold_dataframes_list)
        val_data = k_fold_dataframes_list[0]['val']  # the validation set is always the same

        df_train_list = [split['train'] for split in k_fold_dataframes_list]
        
        # over/under sampling of the training set
        if(config['data']['equalizer']['isEnabled']):
            df_train_list = [label_equalizer(train_i_df, config) for train_i_df in df_train_list]

        train_loaders_list = [make_dataloader(config, df_train, train_transforms, validation_mode=False) for df_train in df_train_list]
        metadata = train_loaders_list[0][1] # get metadata from the first train_loader
        train_loaders = [loader for loader, _ in train_loaders_list]  # list of dataloaders for each training set
    else:
        dataset_split_dict = k_fold_dataframes_list[0]  # the first split is used for training and validation
        train_data, val_data = dataset_split_dict['train'], dataset_split_dict['val']

        train_loader, metadata = make_dataloader(config, train_data, train_transforms, validation_mode=False)
        train_loaders = [train_loader]

    val_loader, _ = make_dataloader(config, val_data, val_transforms, validation_mode=True)

    if UQ_method == "TTA":
        test_set_transforms = train_transforms
    else:
        test_set_transforms = val_transforms
    test_loader, _ = make_dataloader(config, df_test, test_set_transforms, validation_mode=True)

    return train_loaders, val_loader, test_loader, metadata



