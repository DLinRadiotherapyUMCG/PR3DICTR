from torch import nn

from torch.optim import Adam


def get_optimizer(config, model):
    """
    Get the optimizer.
    :param config:
    :param model:
    :return: Optimizer
    """
    if config['training']['optimizer']['name'] == 'Adam':
        return Adam(model.parameters(), lr=config['training']['optimizer']['learning_rate'],
                         weight_decay=config['training']['optimizer']['weight_decay'])
    else:
        raise ValueError(f"Optimizer {config['training']['optimizer']['name']} not supported.")
