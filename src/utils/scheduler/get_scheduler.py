import torch
from torch.optim import Optimizer



def get_scheduler(config, optimizer):
    """
    Get the scheduler, for model training.
    Args:
        config (dict): Configuration dictionary
        optimizer (Pytorch optimizer): Pytorch optimizer for which to create the scheduler
    Returns:
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
    """
    if config['training']['scheduler']['name'] == 'cosine':
        # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                    T_0=config['training']['scheduler']['T_0'],
                                                                    T_mult=config['training']['scheduler'][
                                                                        'T_mult'],
                                                                    eta_min=config['training']['scheduler'][
                                                                        'eta_min'])
    
    # TODO: add more options here

    else:
        raise ValueError(f"Scheduler {config['training']['scheduler']['name']} not supported.")
