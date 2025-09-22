from torch.optim import Adam, AdamW, SGD, RMSprop

from src.utils.optimizer.AdaBound import AdaBound

def get_optimizer(config, model):
    """
    Gets the optimizer, for model training.
    Args:
        config: configuration dictionary
        model: PyTorch model to be optimized
    Returns:
        optimizer: PyTorch optimizer
    """
    optimizer = None
    if config['training']['optimizer']['name'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config['training']['optimizer']['learning_rate'],
                         weight_decay=config['training']['optimizer']['weight_decay'])
    elif config['training']['optimizer']['name'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=config['training']['optimizer']['learning_rate'],
                          weight_decay=config['training']['optimizer']['weight_decay'])
    elif config['training']['optimizer']['name'] == 'AdaBound':
        optimizer = AdaBound(model.parameters(), lr=config['training']['optimizer']['learning_rate'],
                             final_lr=config['training']['optimizer']['learning_rate'], weight_decay=config['training']['optimizer']['weight_decay'])
    elif config['training']['optimizer']['name'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=config['training']['optimizer']['learning_rate'],
                weight_decay=config['training']['optimizer']['weight_decay'], momentum=config['training']['optimizer']['momentum'])
    elif config['training']['optimizer']['name'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=config['training']['optimizer']['learning_rate'],
                            weight_decay=config['training']['optimizer']['weight_decay'], momentum=config['training']['optimizer']['momentum'])
    else:
        raise ValueError(f"Optimizer {config['training']['optimizer']['name']} not supported.")

    return optimizer

