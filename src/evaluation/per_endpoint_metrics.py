import torch
import numpy as np

from src.evaluation.metrics.utils import remove_missing


def calculate_metric_for_multiple_endpoints(config: dict, y_pred_list_dict: dict, y_true_list_dict: dict, metric_functions_dict, sample_weights=None):
    """
    Calculate a metric for each endpoint and return a dictionary with the results (as well as the mean metric value over all endpoints).
    
    Args:
        config (dict): config dictionary
        y_pred_list_dict (dict of lists): dict of lists of PyTorch tensors
        y_true_list_dict (dict of lists): dict of lists of PyTorch tensors
        metric_functions_dict (dict): a dictionary with desired endpoint types as the keys, and metric functions as values
    
    Returns:
        mean_metric_value (float): mean metric value over all endpoints
        results_dict (dict): dictionary with the results
    """

    endpoint_list = y_pred_list_dict.keys()
    endpoint_types_list = metric_functions_dict.keys()  # assuming metric_functions_dict is a dict with endpoint types as keys

    results_dict = {}

    for endpoint_name, endpoint_type in zip(endpoint_list, endpoint_types_list):

        # get the metric function for this type of endpoint
        metric_function = metric_functions_dict[endpoint_type]

        # mask out missing values
        labels, preds  = remove_missing(config, np.array(y_true_list_dict[endpoint_name]), np.array(y_pred_list_dict[endpoint_name]))
        
        # calculate the metric
        if sample_weights is not None:
            m_val = metric_function(config, labels, preds, sample_weights=sample_weights)
        else:
            m_val = metric_function(config, labels, preds)
        results_dict[endpoint_name] = round(m_val, 3)
    
    mean_metric_value = round(np.mean(list(results_dict.values())), 3)

    return mean_metric_value, results_dict

