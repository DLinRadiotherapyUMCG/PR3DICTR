import logging
import os

import torch

def save_model(config, model, model_path):
    """
    Save the model to the output directory.
    :param config:
    :param model:
    :param model_path:
    :return:
    """
    logging.info(f'Saving model to {os.path.join(config["paths"]["output"])}{model_path}')
    torch.save(model.state_dict(), os.path.join(config["paths"]["output"], model_path))