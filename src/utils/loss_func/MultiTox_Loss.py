import torch
from torch import nn

from src.constants import MISSING_DATA_VALUE

class MultiTox_Loss(nn.Module):
    """
    A class representing the loss function for multi toxicity prediction.
    accepts a loss function (BCE, ASL, focal, etc.) and applies it to the output and labels dictionaries (i.e. it applies a loss function to a multi-tox problem)
    """
    def __init__(self, config, loss_function) -> None:
        super(MultiTox_Loss, self).__init__()

        self.config = config
        self.endpoints_list = config['columns']['labels']
        #self.valid_endpoints_as_tensor = torch.tensor([0, 1])  # these are the possible labels, anything else (e.g. -1 for missing values) is masked out

        self.missing_endpoints_as_tensor = torch.tensor([MISSING_DATA_VALUE])

        self.loss_function = loss_function



    def forward(self, outputs_dict, labels_dict):
        # stack all of the predictions into a single tensor
        predictions = torch.stack(list(outputs_dict.values()), dim=1).type(torch.float32).squeeze(-1) # transposed! so that num columns = num toxicities
        # targets = torch.stack(list(labels_dict.values()), dim=1).type(torch.float32)

        targets = labels_dict.squeeze(1)
        
        predictions = torch.reshape(predictions, targets.shape).to(predictions.dtype)
        
        # calculate the loss
        batch_loss = self.loss_function(predictions, targets)

        # mask out missing endpoints
        mask = (targets != self.missing_endpoints_as_tensor[0]) # & (targets <= self.valid_endpoints_as_tensor[1])
        batch_loss = torch.nan_to_num(batch_loss, nan=0.0)
        batch_loss *= mask
        
        num_valid_labels = mask.sum() # the number of non-missing labels

        if num_valid_labels == 0: # in case all labels are missing, the loss will be `nan`, so we return 0 instead
            zero_tensor = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)  
            batch_loss_mean = zero_tensor
            batch_loss_dict = {endpoint:zero_tensor for endpoint in self.config['columns']['labels']}
        
        else:
            # mean loss per endpoint
            endpoint_list = self.config['columns']['labels']
            batch_loss_dict = {}
            if len(endpoint_list) > 1:  # if there are multiple endpoints
                for idx, endpoint in enumerate(endpoint_list):
                    batch_loss_dict[endpoint] = batch_loss[:, idx].sum() / mask[:, idx].sum()
            else:            # if there is only one endpoint
                batch_loss_dict[endpoint_list[0]] = batch_loss[:].sum() / mask[:].sum()            

            # total mean loss
            batch_loss_mean = batch_loss.sum() / mask.sum()
            batch_loss_mean = torch.clamp(batch_loss_mean, min=0, max=10000) 

        
        return batch_loss_mean, batch_loss_dict



