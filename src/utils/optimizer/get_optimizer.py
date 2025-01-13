from torch.optim import Adam, AdamW, SGD, RMSprop

from src.utils.optimizer.AdaBound import AdaBound

def get_optimizer(config, model):
    """
    Get the optimizer.
    :param config:
    :param model:
    :return: Optimizer
    """
    option = None
    if config['training']['optimizer']['name'] == 'Adam':
        option = Adam(model.parameters(), lr=config['training']['optimizer']['learning_rate'],
                         weight_decay=config['training']['optimizer']['weight_decay'])
    elif config['training']['optimizer']['name'] == 'AdamW':
        option = AdamW(model.parameters(), lr=config['training']['optimizer']['learning_rate'], 
                 weight_decay=config['training']['optimizer']['weight_decay'])
    elif config['training']['optimizer']['name'] == 'AdaBound':
        option = AdaBound(model.parameters(), lr=config['training']['optimizer']['learning_rate'], 
                 final_lr=config['training']['optimizer']['learning_rate'], weight_decay=config['training']['optimizer']['weight_decay'])
    elif config['training']['optimizer']['name'] == 'SGD':
        option = SGD(model.parameters(), lr=config['training']['optimizer']['learning_rate'], 
                weight_decay=config['training']['optimizer']['weight_decay'], momentum=config['training']['optimizer']['momentum'])
    elif config['training']['optimizer']['name'] == 'RMSprop':
        option = RMSprop(model.parameters(), lr=config['training']['optimizer']['learning_rate'], 
                weight_decay=config['training']['optimizer']['weight_decay'], momentum=config['training']['optimizer']['momentum'])
    else:
        raise ValueError(f"Optimizer {config['training']['optimizer']['name']} not supported.")
    return option
    
