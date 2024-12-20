import torch
from typing import Tuple





def move_batch_to_device(batch : dict, device : torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Moves the batch (from the dataloader) to the device (CPU or GPU) for the model input and labels.

    Args:
        batch (dict): a dictionary containing the input, clinical features and targets (i.e. a dataloader batch)
        device (torch.device or string): the device to move the data to
    
    Returns:
        inputs (torch.Tensor): the input data
        clinical_features (torch.Tensor): the clinical features
        targets (torch.Tensor): the targets
    """
    # batch is a dictionary, retrieve the input, clinical features and targets
    inputs, clinical_features, targets = batch['input'], batch['features'], batch['label_list']

    # send to device
    inputs = inputs.to(device=device)
    clinical_features = clinical_features.to(device=device)
    targets = targets.to(device=device)

    return inputs, clinical_features, targets