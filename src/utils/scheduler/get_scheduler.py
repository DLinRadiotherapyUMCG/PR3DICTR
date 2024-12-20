import torch
from torch.optim import Optimizer



def get_scheduler(config, optimizer):
    """
    Get the scheduler.
    :param config:
    :param optimizer:
    :return: Scheduler
    """
    if config['training']['scheduler']['name'] == 'cosine':
        # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                    T_0=config['training']['scheduler']['T_0'],
                                                                    T_mult=config['training']['scheduler'][
                                                                        'T_mult'],
                                                                    eta_min=config['training']['scheduler'][
                                                                        'eta_min'])
    else:
        raise ValueError(f"Scheduler {config['training']['scheduler']['name']} not supported.")
