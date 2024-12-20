import os

import yaml


def load_config(config_path):
    """
    Load the config file for the toxicity.
    :param tox:
    :return: Config
    """
    
    #config_path = 'src/Config_presets/' + name + '.yaml'

    print(config_path)

    # Check if the config file exists
    if not os.path.exists(config_path):
        raise ValueError('Config file not found. Please check the toxicity_configs folder.')

    # Load the config file
    with open(config_path, 'r') as f:
        imported_config = yaml.safe_load(f)

    return imported_config

