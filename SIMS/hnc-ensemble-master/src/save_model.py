import logging
import os

import torch
from src.utils.fileHandler import create_folder

def save_model(config, model, model_path):
    """
    Save the model to the output directory.
    :param config:
    :param model:
    :param model_path:
    :return:
    """
    pathToSave = os.path.join(config["paths"]["output"],config["hyperparam_tuning"]["ProjectName"])
    fileLocation = os.path.join(pathToSave, model_path)

    # Create folder if does not exist
    create_folder(fileLocation)

    # Log and Save
    logging.info(f'Saving model to {fileLocation}')
    torch.save(model.state_dict(), pathToSave, model_path)