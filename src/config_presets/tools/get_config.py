# -*- coding: utf-8 -*-
import logging
import yaml
import os

from src.config_presets.tools.load_config import load_config

from src.dataset.LabelTypesManager import LabelTypesManager


def get_config(name, name_base = 'Base_config', pathGiven = ""):
    """
    Constructs config as used throughout the main file 
    """
    
    # Load 
    base_config = load_config(name_base,pathGiven)
    experiment_config = load_config(name,pathGiven)
    
    # Initiate the to be returned config file
    config = base_config.copy()

    update_config(config, experiment_config)

    # if testMode is active, overwrite the settings used in the testMode_config.yaml file
    if config['general']['testMode']:
        testMode_config = load_config('testMode_config', pathGiven)
        update_config(config, testMode_config, allow_new_keys=True)

    # check the config for errors
    check_config(config)
    
    return config




def check_config(config):
    """
    Check label-related settings in the config file.
    """
    assert config['columns']['labels'] is not None, "Labels must be specified in the config file."

    # if labels_types is a string, we can assume that all labels have the same type
    if type(config['columns']['labels_types']) is str:
        # if the labels_types is a string, convert it to a list of the same length as the labels list
        config['columns']['labels_types'] = [config['columns']['labels_types']] * len(config['columns']['labels'])
    
    assert len(config['columns']['labels']) == len(config['columns']['labels_types']), "Number of labels must match number of label types."

    # if the label_column_names is not specified, derive them from the labels and label_types in the current confg
    if "label_column_names" not in config['saving'].keys(): 
        labelManager = LabelTypesManager(config=config)  # get the label types manager from the config
        config['saving']['label_column_names'] = labelManager.label_names_full_list




def update_config(base, updates, allow_new_keys=False):
    """
    Recursively updates the base dictionary with values from the updates dictionary.
    Only non-dictionary values are overwritten. \
    (i.e. if only one thing in the 'model' key needs to be updated, the rest of the 'model' dict will remain the same)
    """
    for key, value in updates.items():
        if key == 'hyperparam_tuning': #  override option to allow `hyperparam_tuning` key to be added to the config
            allow_new_keys = True      #  often, this is not in the base config (because then it would need to contain every possible hyperparameter tuning variable)
        
        if (key in base) and (isinstance(base[key], dict)) and (isinstance(value, dict)):  # if we are dealing with nested dictionaries
            update_config(base[key], value, allow_new_keys=allow_new_keys)
        else:          
            if (key not in base) and (not allow_new_keys):      # if the new config has entries that do not exist in the base config, raise an error
                raise KeyError(f"Key '{key}' not found in base dictionary.")
            else:
                base[key] = value                              # update the base config with the new value for this entry
