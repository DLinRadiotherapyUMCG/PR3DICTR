import torch
from torch import nn

import functools

from src.constants import DEVICE

import torch

from src.utils.loss_func.loss_Focal import FocalLoss
from src.utils.loss_func.loss_Hill import Hill
from src.utils.loss_func.loss_ASL import AsymmetricLossOptimized
from src.utils.loss_func.loss_NegativeLog import NegativeLogLikelihood
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
    if(config['training']['loss']['name'] == 'BCE'):
        loss_function = torch.nn.BCEWithLogitsLoss(reduction = reduction, pos_weight = weights)
    elif(config['training']['loss']['name'] =='softmargin'):       
        loss_function = torch.nn.MultiLabelSoftMarginLoss(reduction = reduction, weight = weights)  # BUG: does not work properly, the outputted dimensions are incorrect
    elif(config['training']['loss']['name'] =='focal'):
        alpha = -1
        if(config['training']['loss']['compensate_imbalance']):
            alpha = config['general']['classImbalance']

        loss_function = FocalLoss(alpha = alpha, reduction=reduction)    #TODO: add gamma and alpha parameters to configs
    elif(config['training']['loss']['name'] == 'hill'):
        loss_function = Hill(reduction = reduction) # TODO: add gamma, margin, lamb parameters to configs
    elif(config['training']['loss']['name'] == "ASL"):
        loss_function = AsymmetricLossOptimized(reduction = reduction) # TODO: add gamma_neg and gamma_pos parameters to configs
    elif(config['training']['loss']['name'] == 'NLLL'):
        # Loss function for time event --> Requires 2 inputs (days, binaryCheckEvent)
        if(len(config['columns']['label']) != 2):
            raise Exception(f"Error: NLLL loss function expects 2 labels and got {config['columns']['label']}, abort code.")
        loss_function = NegativeLogLikelihood(config)
    
    else:
        raise ValueError(f"Loss function {config['training']['loss']['name']} not supported.")
    
    # wrap the loss function into a handler (behaves as a normal loss function in the training code)
    multitox_loss_handler = MultiTox_Loss(config, loss_function)

    return multitox_loss_handler
    

