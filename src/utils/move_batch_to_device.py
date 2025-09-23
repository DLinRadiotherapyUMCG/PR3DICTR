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

    # send to device
    if 'input' in batch.keys():
        inputs = batch['input'].to(device=device, non_blocking=True)
    else:
        inputs = None
    clinical_features = batch['features'].to(device=device, non_blocking=True)
    targets = batch['label_list'].to(device=device, non_blocking=True)

    return inputs, clinical_features, targets