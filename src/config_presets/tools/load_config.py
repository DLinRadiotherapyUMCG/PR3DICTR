import os

import yaml
import logging


def load_config(name, pathGiven = ""):
    """
    Load the config file for the toxicity.
    :param tox:
    :return: Config
    """

    if not name.endswith('.yaml'):
        name += '.yaml'
    
    if pathGiven == "":
        config_path = os.path.join("src", "config_presets", name) 
    else:
        config_path = os.path.join(pathGiven, "src", "config_presets", name) 

    # Check if the config file exists
    if not os.path.exists(config_path):
        raise ValueError('Config file not found. Please check the toxicity_configs folder.')

    # Load the config file
    with open(config_path, 'r') as f:
        imported_config = yaml.load(f, Loader=yaml.FullLoader)

    return imported_config
