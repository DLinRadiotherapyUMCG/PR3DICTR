import logging


def check_improvement(config, val_loss, val_mean_metric_value, best_value):
    """
    Function to check whether the model has improved on the current epoch. Does so using either the main metric, or the loss value.
    Args: 
        config (dict):
        val_loss : a loss value
        val_mean_metric_value: a metric value
        best_value: the running best (loss or metric) value
    Returns:
        best_value: the (new) running best (loss or metric) value
        improved (bool): a boolean indicating if this epoch was better than the previous best epoch
    """
    improved = False
    # if otpimising the mean loss
    if config['training']['stopping_criteria'] == "loss":
        if val_loss < best_value:
            logging.info(f'New lowest loss: {val_loss}')
            best_value = val_loss
            improved = True

    # if optimising the mean metric (e.g. validation AUC) # NOTE: DANIEL TODO:  maybe needs to be changed if some metrics are to be minimised?
    else:
        if val_mean_metric_value > best_value:
            logging.info(f'New highest {config["general"]["metric_name"]}: {val_mean_metric_value}')
            best_value = val_mean_metric_value
            improved = True

    return best_value, improved