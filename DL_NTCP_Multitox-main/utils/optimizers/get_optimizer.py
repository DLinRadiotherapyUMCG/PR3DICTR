import torch
import utils.optimizers as optimizers


def get_optimizer(config, model, num_batches_per_epoch, logger) -> torch.optim.Optimizer:
    """
    Function to initialise the optimizer. Relies primarily on the config object's information 
    to retrieve the correct parameters for the optimizer.
    https://github.com/jettify/pytorch-optimizer

    Args:
        config:
        optimizer_name:
        model: 
        lr: 
        num_batches_per_epoch: 
        logger:

    Returns:
        optimizer: 

    """
    optimizer_name = config.optimizer_name
    lr = config.lr

    momentum = config.momentum
    weight_decay = config.weight_decay
    hessian_power = config.hessian_power

    logger.my_print('Loading optimizer: {}.'.format(optimizer_name))

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam_w':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'sgd_w':
        optimizer = optimizers.SGDW(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'ada_hessian':
        optimizer = optimizers.Adahessian(model.parameters(), lr=lr, weight_decay=weight_decay,
                                          hessian_power=hessian_power)
    elif optimizer_name == 'acc_sgd':
        optimizer = optimizers.AccSGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'ada_belief':
        optimizer = optimizers.AdaBelief(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'ada_bound':
        optimizer = optimizers.AdaBound(model.parameters(), lr=lr, final_lr=lr * 100, weight_decay=weight_decay)
    elif optimizer_name == 'ada_bound_w':
        optimizer = optimizers.AdaBoundW(model.parameters(), lr=lr, final_lr=lr * 100, weight_decay=weight_decay)
    elif optimizer_name == 'ada_mod':
        optimizer = optimizers.AdaMod(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'apollo':
        optimizer = optimizers.Apollo(model.parameters(), lr=lr, init_lr=lr / 100, warmup=500,
                                      weight_decay=weight_decay)
    elif optimizer_name == 'diff_grad':
        optimizer = optimizers.DiffGrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'madgrad':
        optimizer = optimizers.MADGRAD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'novo_grad':
        optimizer = optimizers.NovoGrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'pid':
        optimizer = optimizers.PID(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'qh_adam':
        optimizer = optimizers.QHAdam(model.parameters(), lr=lr, nus=(0.7, 1.0), betas=(0.995, 0.999),
                                      weight_decay=weight_decay)
    elif optimizer_name == 'qhm':
        optimizer = optimizers.QHM(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'r_adam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'ranger_qh':
        optimizer = optimizers.RangerQH(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'ranger_21':
        optimizer = optimizers.Ranger21(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                                        num_epochs=150, num_batches_per_epoch=num_batches_per_epoch)
    elif optimizer_name == 'swats':
        optimizer = optimizers.SWATS(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'yogi':
        optimizer = optimizers.Yogi(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer_name: {}.'.format(optimizer_name))

    if config.use_lookahead:
        return optimizers.Lookahead(optimizer, k=config.lookahead_k, alpha=config.lookahead_alpha)

    return optimizer
