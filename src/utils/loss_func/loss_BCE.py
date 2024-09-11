import torch
from torch import nn

import functools

from src.constants import DEVICE

import torch

def multi_BCEloss(config, outputs_dict, labels_dict):

    predictions = torch.stack(list(outputs_dict.values()), dim=1).type(torch.float32) # transposed! so that num columns = num toxicities
    
    # print(predictions.shape)
    # print(labels_dict)
    
    # targets = torch.stack(list(labels_dict.values()), dim=1).type(torch.float32)
    valid_endpoints_as_tensor = torch.tensor([0,1])
    
    targets = labels_dict
    
    predictions = torch.reshape(predictions, targets.shape).to(predictions.dtype)
    
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(config['training']['loss']['weight'],  dtype=torch.float32, device=DEVICE))
    batch_loss = loss_function(predictions, targets)

    mask = (targets >= valid_endpoints_as_tensor[0]) & (targets <= valid_endpoints_as_tensor[1])

    # print(batch_loss)
    batch_loss = torch.nan_to_num(batch_loss, nan=0.0)
    batch_loss *= mask
    

    batch_loss_mean = batch_loss.sum() / mask.sum()

    #print(batch_loss_mean)
    batch_loss_mean = torch.clamp(batch_loss_mean, min=0, max=10000) 

    return batch_loss_mean