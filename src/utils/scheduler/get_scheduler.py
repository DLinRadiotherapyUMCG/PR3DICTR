import torch

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
                                                                    T_0=config['training']['scheduler']['cosine']['T_0'],
                                                                    T_mult=config['training']['scheduler']['cosine']['T_mult'],
                                                                    eta_min=config['training']['scheduler']['cosine']['eta_min'])

    elif config['training']['scheduler']['name'] == 'step':
        # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=config['training']['scheduler']['step']['step_size'],
                                               gamma=config['training']['scheduler']['step']['gamma'])

    elif config['training']['scheduler']['name'] == 'plateau':
        # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                          mode=config['training']['scheduler']['plateau']['mode'],
                                                          factor=config['training']['scheduler']['plateau']['factor'],
                                                          patience=config['training']['scheduler']['plateau']['patience'],
                                                          min_lr=config['training']['scheduler']['plateau']['min_lr'])

    else:
        raise ValueError(f"Scheduler {config['training']['scheduler']['name']} not supported.")
