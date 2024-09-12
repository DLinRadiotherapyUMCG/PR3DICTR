# -*- coding: utf-8 -*-
import os
from src.utils.get_encoder import get_encoder
from src.models.linear_layers import MultiToxOutputHead

from torchinfo import summary
from src.utils.fileHandler import create_folder


def get_model_summary(config, model, input_size, device, save_to_file = True):
    
    """
    Get model summary and number of trainable parameters.

    Args:
        model:
        input_size:
        device:
        logger:

    Returns:
        total_params (int): number of trainable parameters

    """
    # Get and save summary
    verbose = 0
    txt = str(summary(model=model, input_size=input_size, device=device, verbose=verbose, depth=5,
                      col_names=["input_size","output_size","num_params", "kernel_size"], row_settings=["var_names"]))
    if save_to_file:
        fileLocation = os.path.join(os.path.join(os.path.join(config["paths"]["output"],config["hyperparam_tuning"]["ProjectName"]),config["general"]["trialNumber"]),config['Save']['fileNames']['modelSummary'])   
        create_folder(fileLocation)
        file = open(fileLocation, 'a+', encoding='utf-8')
        file.write(txt)
        file.close()

    # Determine number of trainable parameters
    # Source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    # total_params = sum(p.numel() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params