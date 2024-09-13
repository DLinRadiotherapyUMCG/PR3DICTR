# -*- coding: utf-8 -*-

# from load_config import load_config

import yaml
import os

def load_config(name, pathGiven = ""):
    """
    Load the config file for the toxicity.
    :param tox:
    :return: Config
    """
    
    if(pathGiven == ""):
        config_path = 'src/config_presets/' + name + '.yaml'
    else:
        config_path = pathGiven + '\\' + 'src\\config_presets\\' + name + '.yaml'

    print(config_path)

    # Check if the config file exists
    if not os.path.exists(config_path):
        raise ValueError('Config file not found. Please check the toxicity_configs folder.')

    # Load the config file
    with open(config_path, 'r') as f:
        imported_config = yaml.safe_load(f)

    return imported_config


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
        update_config(config, testMode_config)
    
    return config


def update_config(base, updates):
    """
    Recursively updates the base dictionary with values from the updates dictionary.
    Only non-dictionary values are overwritten. \
    (i.e. if only one thing in the 'model' key needs to be updated, the rest of the 'model' dict will remain the same)
    """
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            update_config(base[key], value)
        else:
            base[key] = value

    # Example usage:
    # base_config = {'a': 1, 'b': {'c': 2, 'd': 3}}
    # updates = {'b': {'c': 4}, 'e': 5}
    # update_config(base_config, updates)
    # print(base_config)  # Output: {'a': 1, 'b': {'c': 4, 'd': 3}, 'e': 5}