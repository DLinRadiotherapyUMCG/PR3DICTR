import os

# import yaml
import logging
from ruamel.yaml import YAML


# yaml.indent(mapping=4, sequence=4, offset=2)
# yaml.width = 80
# yaml.preserve_quotes = True

def load_config(name, pathGiven = "") -> dict:
    """
    Load the config file for the toxicity.
    Args:
      name (str or path): The name of the config file (without .yaml extension)
      pathGiven (str): The path to the directory containing the config file (optional)
    Returns:
      config (dict): The loaded config as a dictionary
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
    yaml = YAML()
    # yaml.width = 80
    yaml.preserve_quotes = True

    with open(config_path) as f:
        imported_config = yaml.load(f)

    # with open(config_path, 'r') as f:
    #     imported_config = yaml.load(f, Loader=yaml.FullLoader)

    return imported_config
