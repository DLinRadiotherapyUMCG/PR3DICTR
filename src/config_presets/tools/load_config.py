import os

import yaml
import logging


def load_config(name, pathGiven = ""):
    """
    Load the config file for the toxicity.
    :param tox:
    :return: Config
    """
    
    if pathGiven == "":
        #config_path = 'src/config_presets/' + name + '.yaml'
        config_path = os.path.join("src", "config_presets", name + ".yaml")
    else:
        #config_path = pathGiven + '\\' + 'src\\config_presets\\' + name + '.yaml'
        config_path = os.path.join(pathGiven, "src", "config_presets", name + ".yaml")

    print(config_path)

    # Check if the config file exists
    if not os.path.exists(config_path):
        raise ValueError('Config file not found. Please check the toxicity_configs folder.')

    # Load the config file
    with open(config_path, 'r') as f:
        imported_config = yaml.safe_load(f)

    return imported_config
