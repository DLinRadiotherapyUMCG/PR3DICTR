import torch
from torch import nn

import functools

from src.constants import DEVICE

import torch

from src.utils.loss_func.loss_Focal import FocalLoss
from src.utils.loss_func.loss_Hill import Hill
from src.utils.loss_func.loss_ASL import AsymmetricLossOptimized
# from pred_RT.graveyard.loss_NegativeLog import NegativeLogLikelihood
from src.utils.loss_func.MultiTox_Loss import MultiTox_Loss
from src.utils.loss_func.MultiLabel_Loss import MultiLabel_Loss


def get_loss_function(config : dict, LabelTypesManager):
    """
    Retrieves the loss function.
    Args:
        config (dict): the configuration dict
    Returns:
        loss_function (torch.nn.Module): the loss function
    """
    
    #weights = None
    reduction = config['training']['loss']['reduction']
    loss_name = config['training']['loss']['name'].lower() # make the loss function name lowercase

    if loss_name == 'bce':
        pos_weights = torch.as_tensor(config['training']['loss']['BCE']['pos_weight'], dtype=torch.float32, device=DEVICE)
        loss_function = torch.nn.BCEWithLogitsLoss(reduction = reduction, pos_weight = pos_weights)

    # elif(loss_name == 'softmargin'):       
    #     loss_function = torch.nn.MultiLabelSoftMarginLoss(reduction = reduction, weight = weights)  # BUG: does not work properly, the outputted dimensions are incorrect

    elif loss_name == 'focal':
        alpha = config['training']['loss']['focal']['alpha']
        gamma = config['training']['loss']['focal']['gamma']
        loss_function = FocalLoss(alpha = alpha, gamma = gamma, reduction=reduction)    

    elif loss_name == 'hill':
        loss_function = Hill(reduction = reduction) # TODO: add gamma, margin, lamb parameters to configs

    elif loss_name == "asl":
        gamma_neg = config['training']['loss']['ASL']['gamma_neg']
        gamma_pos = config['training']['loss']['ASL']['gamma_pos']
        loss_function = AsymmetricLossOptimized(gamma_neg=gamma_neg, gamma_pos=gamma_pos, reduction = reduction)

    # elif loss_name == 'nlll':
    #     # Loss function for time event --> Requires 2 inputs (days, binaryCheckEvent)
    #     if(len(config['columns']['label']) != 2):
    #         raise Exception(f"Error: NLLL loss function expects 2 labels and got {config['columns']['label']}, abort code.")
    #     loss_function = NegativeLogLikelihood(config)
    
    else:
        raise ValueError(f"Loss function {loss_name} not supported.")
    
    # wrap the loss function into a handler (behaves as a normal loss function in the training code)
    multitox_loss_handler = MultiLabel_Loss(config, binary_loss_function=loss_function, LabelTypesManager=LabelTypesManager)

    return multitox_loss_handler
    

