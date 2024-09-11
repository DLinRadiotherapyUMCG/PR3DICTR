import torch
from torch import nn

import functools

from src.constants import DEVICE

import torch

from src.utils.loss_func.loss_Focal import FocalLoss
from src.utils.loss_func.loss_Hill import Hill
from src.utils.loss_func.loss_ASL import AsymmetricLossOptimized

from src.utils.loss_func.MultiTox_Loss import MultiTox_Loss


def get_loss_function(config):
    """
    Get the loss function.
    :param config:
    :return: Loss function
    """
    weights = torch.as_tensor(config['training']['loss']['weight'], dtype=torch.float32, device=DEVICE)
    #print("WEIGHTS", weights)
    weights = None
    reduction = config['training']['loss']['reduction']
    match(config['training']['loss']['name']):
        case 'BCE':
            loss_function = torch.nn.BCEWithLogitsLoss(reduction = reduction, pos_weight = weights)
        case 'softmargin':       
            loss_function = torch.nn.MultiLabelSoftMarginLoss(reduction = reduction, weight = weights)  # BUG: does not work properly, the outputted dimensions are incorrect
        case 'focal':
            loss_function = FocalLoss(reduction=reduction)    #TODO: add gamma and alpha parameters to configs
        case 'hill':
            loss_function = Hill(reduction = reduction) # TODO: add gamma, margin, lamb parameters to configs
        case "ASL":
            loss_function = AsymmetricLossOptimized(reduction = reduction) # TODO: add gamma_neg and gamma_pos parameters to configs
        case _:
            raise ValueError(f"Loss function {config['training']['loss']['name']} not supported.")
    
    # wrap the loss function into a handler (behaves as a normal loss function in the training code)
    multitox_loss_handler = MultiTox_Loss(config, loss_function)

    return multitox_loss_handler
    

