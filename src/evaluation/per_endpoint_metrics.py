import torch
import numpy as np

from src.evaluation.metrics.utils import remove_missing


def calculate_metric_for_multiple_endpoints(config: dict, y_pred_list_dict: dict, y_true_list_dict: dict, metric_function, sample_weights=None):
    """
    Calculate a metric for each endpoint and return a dictionary with the results (as well as the mean metric value over all endpoints).
    
    Args:
        config (dict): config dictionary
        y_pred_list_dict (dict of lists): dict of lists of PyTorch tensors
        y_true_list_dict (dict of lists): dict of lists of PyTorch tensors
        metric_function (function): function that calculates the metric
    
    Returns:
        mean_metric_value (float): mean metric value over all endpoints
        results_dict (dict): dictionary with the results
    """
    endpoint_list = config['columns']['labels']

    results_dict = {}

    # temp dicts to store the predictions and labels for each endpoint
    predictions_dict = {}
    labels_dict = {}
    
    for endpoint in endpoint_list:
        # mask out missing values
        labels, preds  = remove_missing(config, np.array(y_true_list_dict[endpoint]), np.array(y_pred_list_dict[endpoint]))
        
        predictions_dict[endpoint] = preds
        labels_dict[endpoint] = labels

        # calculate the metric
        if sample_weights is not None:
            m_val = metric_function(config, labels_dict[endpoint], predictions_dict[endpoint], sample_weights=sample_weights)
        else:
            m_val = metric_function(config, labels_dict[endpoint], predictions_dict[endpoint])
        results_dict[endpoint] = round(m_val, 3)
    
    mean_metric_value = round(np.mean(list(results_dict.values())), 3)

    return mean_metric_value, results_dict

