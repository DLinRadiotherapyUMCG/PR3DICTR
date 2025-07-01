




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
    
    # Return the modified config
    #return config
