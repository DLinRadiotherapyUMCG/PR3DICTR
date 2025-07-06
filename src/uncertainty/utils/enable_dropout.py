import torch


def enable_dropout(model):
    """
    Enable dropout layers in the model for Monte Carlo Dropout.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()  # Set dropout layers to training mode
    return model