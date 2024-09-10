# -*- coding: utf-8 -*-

# from load_config import load_config

import yaml
import os

def load_config(name):
    """
    Load the config file for the toxicity.
    :param tox:
    :return: Config
    """
    
    config_path = 'src/Config_presets/' + name + '.yaml'

    print(config_path)

    # Check if the config file exists
    if not os.path.exists(config_path):
        raise ValueError('Config file not found. Please check the toxicity_configs folder.')

    # Load the config file
    with open(config_path, 'r') as f:
        imported_config = yaml.safe_load(f)

    return imported_config


def get_config(name, name_base = 'Base_config'):
    """
    Constructs config as used throughout the main file 
    """
    
    
    # Load 
    base_config = load_config(name_base)
    experiment_config = load_config(name)
    
    # Initiate the to be returned config file
    config = base_config.copy()
    
    # Replaces or adds information from the experiment config
    for key in experiment_config.keys():
        config[key] = experiment_config[key]
    
    return config