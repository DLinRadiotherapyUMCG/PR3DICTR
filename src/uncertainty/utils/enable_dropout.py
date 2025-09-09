import torch


def enable_dropout(model):
    """
    Enable dropout layers in the model for Monte Carlo Dropout.
    Args: 
        model: PyTorch model
    Returns:
        model: PyTorch model with dropout layers enabled during inference
    """
    
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()  # Set dropout layers to training mode
    return model