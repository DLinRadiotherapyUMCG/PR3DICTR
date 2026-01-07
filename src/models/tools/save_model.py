import logging
import os
import torch
from src.utils.fileHandler import create_folder


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


