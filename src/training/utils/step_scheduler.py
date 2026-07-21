

def step_scheduler(config, scheduler, epoch_num, batch_num, num_batches_per_epoch):
    """Step scheduler based on configuration"""
    scheduler_name = config['training']['scheduler']['name']

    if scheduler is None:
        return  # No scheduler to step
    
    if scheduler_name == 'cosine':
        scheduler.step((epoch_num - 1) + (batch_num / num_batches_per_epoch))
    elif scheduler_name == 'exponential':
        if batch_num % config['training']['scheduler']['step_size'] == 0:
            scheduler.step()
    elif scheduler_name in ['cyclic']:
        scheduler.step()