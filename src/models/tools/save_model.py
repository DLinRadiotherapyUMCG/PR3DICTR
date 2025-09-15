import logging
import os
import yaml
import pandas as pd
import torch
from src.utils.fileHandler import create_folder, create_file
from src.constants import PATIENT_ID_COL_NAME


def save_model(config, model):
    """
    Save the model to the output directory.
    Args:
        config: configuration dictionary
        model: PyTorch model to be saved
    Returns:
        None
    """
    
    pathToSave = config['general']['resultsCurrentDirectory']
    model_weights_filename = config['saving']['filenames']['model_weights']
    fileLocation = os.path.join(pathToSave, model_weights_filename)

    # Create folder if does not exist
    create_folder(pathToSave)

    # Log and Save
    logging.info(f'Saving model to {fileLocation}')
    torch.save(model.state_dict(), fileLocation)



def load_model(config, model):
    """
    Load the model from the output directory.
    Args:
        config: configuration dictionary
        model: PyTorch model to be loaded
    Returns:
        model: Loaded PyTorch model
    """
    pathToSave = config['general']['resultsCurrentDirectory']
    model_weights_filename = config['saving']['filenames']['model_weights']
    fileLocation = os.path.join(pathToSave, model_weights_filename)

    # Log and Save
    logging.info(f'Loading model from {fileLocation}')
    model.load_state_dict(torch.load(fileLocation, weights_only=True))
    return model



def save_config(config, fileName):
    """
    Save the config as a .yaml file.
    Args:
        config: configuration dictionary
        fileName: name of the file to save the config as (should end with .yaml)
    Returns:
        None
    """

    directoryToSave = config['general']['resultsCurrentDirectory']
    if(fileName.endswith(".yaml") == False):
        fileName = f"{fileName}.yaml"
    pathConfig = os.path.join(directoryToSave,fileName)
    with open(pathConfig,'w') as outfile:
        yaml.dump(config,outfile,default_flow_style=False)


