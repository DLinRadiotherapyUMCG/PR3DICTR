import os
import logging
import torch

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
