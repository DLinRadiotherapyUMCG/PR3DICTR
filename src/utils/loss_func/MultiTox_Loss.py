import torch
from torch import nn



class MultiTox_Loss(nn.Module):
    """
    A class representing the loss function for multi toxicity prediction.
    accepts a loss function (BCE, ASL, focal, etc.) and applies it to the output and labels dictionaries (i.e. it applies a loss function to a multi-tox problem)
    """
    def __init__(self, config, loss_function) -> None:
        super(MultiTox_Loss, self).__init__()

        self.config = config
        self.endpoints_list = config['columns']['label']
        self.valid_endpoints_as_tensor = torch.tensor([0, 1])  # these are the possible labels, anything else (e.g. -1 for missing values) is masked out

        self.loss_function = loss_function

    def forward(self, outputs_dict, labels_dict):
        predictions = torch.stack(list(outputs_dict.values()), dim=1).type(torch.float32).squeeze(-1) # transposed! so that num columns = num toxicities
    
        # targets = torch.stack(list(labels_dict.values()), dim=1).type(torch.float32)


        #valid_endpoints_as_tensor = torch.tensor([0,1])
        
        # TODO: labels_dict should become an actual dictionary, now its just a tensor ?!?!?!?!
        targets = labels_dict

        targets = targets.squeeze(1)
        
        predictions = torch.reshape(predictions, targets.shape).to(predictions.dtype)
       
        # print(targets,predictions)

        batch_loss = self.loss_function(predictions, targets)

        # print(batch_loss)
        mask = (targets >= self.valid_endpoints_as_tensor[0]) & (targets <= self.valid_endpoints_as_tensor[1])

        # print(mask)
        batch_loss = torch.nan_to_num(batch_loss, nan=0.0)
        batch_loss *= mask
        
        num_valid_labels = mask.sum()

        if num_valid_labels == 0: # in case all labels are missing, the loss will be `nan`, so we return 0 instead
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        else:
            batch_loss_mean = batch_loss.sum() / mask.sum()

            #print(batch_loss_mean)
            batch_loss_mean = torch.clamp(batch_loss_mean, min=0, max=10000) 

            return batch_loss_mean



