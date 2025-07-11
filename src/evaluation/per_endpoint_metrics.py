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
    endpoint_names_to_types_dict = {name:type for name, type in zip(config['columns']['labels'], config['columns']['labels_types'])}

    results_dict = {}

    for endpoint_name in endpoint_list:
        endpoint_type = endpoint_names_to_types_dict[endpoint_name]

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

