import torch


def get_scheduler(config, scheduler_name, optimizer, optimal_lr, num_batches_per_epoch, logger):
    scheduler = None

    T_0restart = config.T_0restart
    T_mult = config.T_mult 
    eta_min = config.eta_min
    step_size_up = config.step_size_up
    step_size = config.step_size
    sched_gamma = config.sched_gamma

    if scheduler_name == 'cosine':
        # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=int(T_0restart),
                                                                         T_mult=T_mult,
                                                                         eta_min=eta_min)
    elif scheduler_name == 'cyclic':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=config.base_lr, max_lr=config.max_lr,
                                                      step_size_up=step_size_up, cycle_momentum=False)

    elif scheduler_name == 'exponential':
        # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=sched_gamma)

    elif scheduler_name == 'step':
        # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=sched_gamma)

    return scheduler