import torch
from torch import nn

import functools

from src.constants import DEVICE

import torch

from src.utils.loss_func.loss_softMargin import multi_softmargin_loss
from src.utils.loss_func.loss_BCE import multi_BCEloss
from src.utils.loss_func.loss_Focal import multi_focal
from src.utils.loss_func.loss_Hill import multi_HillLoss

def get_loss_function(config):
    """
    Get the loss function.
    :param config:
    :return: Loss function
    """
    weight = torch.as_tensor(config['training']['loss']['weight'], dtype=torch.float32, device=DEVICE)
    reduction = config['training']['loss']['reduction']
    match(config['training']['loss']['name']):
        case 'BCEloss':
            return multi_BCEloss
        case 'softmargin':
            return multi_softmargin_loss
        case 'focal':
            print("TODO: FOCAL NEEDS TESTING")
            return multi_focal
        case 'hill':
            print("TODO: Hill NEEDS TESTING")
            return multi_HillLoss
        case _:
            raise ValueError(f"Loss function {config['training']['loss']['name']} not supported.")
    

