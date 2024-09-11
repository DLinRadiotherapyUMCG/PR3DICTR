import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: TESTING IF IT WORKS

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight: torch.Tensor | None = None, gamma = 2, reduction: str = 'mean') -> None:
        super().__init__(weight, reduction = reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input,target, reduction=self.reduction, weight = self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1- pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    
def multi_focal(config, outputs_dict, labels_dict):

    predictions = torch.stack(list(outputs_dict.values()), dim=1).type(torch.float32) # transposed! so that num columns = num toxicities
    
    # print(predictions.shape)
    # print(labels_dict)
    
    # targets = torch.stack(list(labels_dict.values()), dim=1).type(torch.float32)
    valid_endpoints_as_tensor = torch.tensor([0,1])
    
    targets = labels_dict
    
    predictions = torch.reshape(predictions, targets.shape).to(predictions.dtype)
    
    loss_function = FocalLoss(weight=torch.tensor(config['training']['loss']['weight'], dtype=torch.float32),reduction='none',) 
    batch_loss = loss_function(predictions, targets)

    mask = (targets >= valid_endpoints_as_tensor[0]) & (targets <= valid_endpoints_as_tensor[1])

    # print(batch_loss)
    batch_loss = torch.nan_to_num(batch_loss, nan=0.0)
    batch_loss *= mask
    
    batch_loss_mean = batch_loss.sum() / mask.sum()

    #print(batch_loss_mean)
    batch_loss_mean = torch.clamp(batch_loss_mean, min=0, max=10000) 

    return batch_loss_mean
