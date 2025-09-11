from src.evaluation.metrics.classification import auc, auc_se, balanced_acc, accuracy, F1_score, precision, recall
from src.evaluation.metrics.regression import r2, MSE, MAE, RMSE
from src.evaluation.metrics.calibration import brier, ECE, MCE, ACE
from src.evaluation.metrics.events import c_index


def get_metric_function(metric_name : str):
    """
    Helper function that retrieves the function used to calculate a certain metric
    Args:
        metric_name (str): name of the metric 
    Returns:
        metric_function: the function that accepts the labels and predictions (and other args) and returns a metric value
    """

    # Classification metrics
    if metric_name == 'AUC':
        return auc 
    elif metric_name == 'AUC_SE':
        return auc_se
    elif metric_name == 'accuracy':
        return accuracy
    elif metric_name == 'balanced_accuracy':
        return balanced_acc
    elif metric_name == 'F1_score':
        return F1_score
    elif metric_name == 'precision':
        return precision
    elif metric_name == 'recall':
        return recall
    
    # Regression metrics
    elif metric_name == 'r2':
        return r2
    elif metric_name == 'MSE':
        return MSE
    elif metric_name == 'MAE':
        return MAE
    elif metric_name == 'RMSE':
        return RMSE

    # Calibration metrics
    elif metric_name == 'brier':
        return brier
    elif metric_name == 'ECE':
        return ECE
    elif metric_name == 'MCE':
        return MCE
    elif metric_name == 'ACE':
        return ACE
    
    # Event-based endpoint metrics
    elif metric_name == 'C-index':
        return c_index

    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

