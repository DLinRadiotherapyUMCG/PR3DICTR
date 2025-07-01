import torch






def enable_dropout(model):
    """
    Enable dropout layers in the model for Monte Carlo Dropout.
    """
    for module in model.modules():
        print(f"Module: {module.__class__.__name__}")
        if isinstance(module, torch.nn.Dropout):
            module.train()  # Set dropout layers to training mode
    return model





def train_MC_dropout_model(config):

    # load the dataset
    pass
