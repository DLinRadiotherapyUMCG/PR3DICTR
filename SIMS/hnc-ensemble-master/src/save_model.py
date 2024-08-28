import logging
import os

import torch
from src.utils.fileHandler import create_folder

def save_model(config, model, modelFile):
    """
    Save the model to the output directory.
    :param config:
    :param model:
    :param model_path:
    :return:
    """
    pathToSave = os.path.join(os.path.join(config["paths"]["output"],config["hyperparam_tuning"]["ProjectName"]),config["general"]["trialNumber"])
    fileLocation = os.path.join(pathToSave, modelFile)

    # Create folder if does not exist
    create_folder(fileLocation)

    # Log and Save
    logging.info(f'Saving model to {fileLocation}')
    torch.save(model.state_dict(), fileLocation)