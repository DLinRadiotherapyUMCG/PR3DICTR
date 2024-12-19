import logging
import copy
from src.models.tools.save_model import save_model


# Check if this model has the lowest validation loss or highest metric

def check_improvement(config, val_loss, val_mean_metric_value, best_value):
    improved = False
    # if otpimising the mean loss
    if config['training']['stopping_criteria'] == "loss":
        if val_loss < best_value:
            logging.info(f'New lowest loss: {val_loss}')
            best_value = val_loss
            improved = True

    # if optimising the mean metric (e.g. validation AUC)
    else:
        if val_mean_metric_value > best_value:
            logging.info(f'New highest {config["general"]["metric_name"]}: {val_mean_metric_value}')
            best_value = val_mean_metric_value
            improved = True

    return best_value, improved