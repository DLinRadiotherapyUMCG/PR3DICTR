"""
Metrics and metric-related functions for evaluating the performance of the models
"""
import numpy as np
from torcheval.metrics import BinaryAUROC
import torch
from sklearn.metrics import roc_auc_score

"""
AUC METRICS
"""

def compute_mixup_AUCs(labels, predictions, mixup_label_indices, mixup_lambda):
    """
    Special function to compute the AUC if MixUp augmentation is used.
    Untangles the mixed labels and computes the AUCs of each one seperately.
    Args:
        labels:
        predictions:
        mixup_label_indices:
        mixup_lambda:
    returns:
        mixup_AUC: an AUC value
    """
    y_a_AUC = roc_auc_score(labels, predictions)
    y_b_AUC = roc_auc_score(labels[mixup_label_indices], predictions)

    mixup_AUC = mixup_lambda * y_a_AUC + (1 - mixup_lambda) * y_b_AUC
    mixup_AUC = mixup_AUC.mean().item()
    
    return mixup_AUC


def compute_AUCs_for_multiple_endpoints(cfg, y_pred_list_dict, y_true_list_dict, mixup_label_indices=None, mixup_lambda=None):
    """
    Computes the AUC for each endpoint.
    Uses two different AUC metrics: BinaryAUROC and ROCAUCMetric because the latter is one-hot encoded (i.e. into 0 and 1 columns).
    Args:
        y_pred_list_dict: A dictionary with the model's predictions for each endpoint, where each value is a list of predictions.
        y_true_list_dict: A dictionary with the true labels for each endpoint, in the same structure as `y_pred_list_dict`.
    Returns:
        avg_auc_value: the average AUC value over all endpoints
        auc_value_dict: a dictionary with the AUC value for each endpoint
    """
    if cfg.loss_function_name == 'mse':
        auc_value_dict = dict()
        for endpoint in cfg.endpoint_list:
            auc_value_dict[endpoint] = 0
        
        return 0, auc_value_dict

    auc_value_dict = dict()
    valid_endpoints_as_tensor = torch.as_tensor(cfg.valid_endpoint_values, device=cfg.device)

    predictions_dict = dict()
    labels_dict = dict()

    for endpoint in cfg.endpoint_list:
        if cfg.num_ohe_classes == 1:
            if type(y_pred_list_dict[endpoint]) == list:
                y_pred_list_dict[endpoint] = torch.tensor(y_pred_list_dict[endpoint], device=cfg.device)

            if type(y_true_list_dict[endpoint]) == list:
                y_true_list_dict[endpoint] = torch.tensor(y_true_list_dict[endpoint], device=cfg.device)

            predictions_dict[endpoint] = y_pred_list_dict[endpoint].clone().detach()
            labels_dict[endpoint] = y_true_list_dict[endpoint].clone().detach()
        elif cfg.num_ohe_classes == 2:
            predictions_dict[endpoint] = torch.tensor([x[1] for x in y_pred_list_dict[endpoint]])
            labels_dict[endpoint] = torch.tensor([x[1] for x in y_true_list_dict[endpoint]])
            #y_true_list_dict[endpoint] = to_onehot(y_true_list_dict[endpoint])

        # remove the missing labels
        mask = torch.isin(y_true_list_dict[endpoint], valid_endpoints_as_tensor) # mask = the labels to keep
        predictions_dict[endpoint] = predictions_dict[endpoint][mask]
        labels_dict[endpoint] = labels_dict[endpoint][mask]

        if cfg.perform_mixup and mixup_label_indices is not None and mixup_lambda is not None:
            auc_value = compute_mixup_AUCs(labels_dict[endpoint].cpu(), predictions_dict[endpoint].cpu(), mixup_label_indices.cpu(), mixup_lambda)
        else:
            auc_value = roc_auc_score(labels_dict[endpoint].cpu(), predictions_dict[endpoint].cpu())
        
        # if in test mode, then its quite likely that a none value is returned,
        # for instance if the val or test set is small and all labels = 0
        # override it with -1, to prevent errors in the rest of the code (this is purely for test mode)
        if cfg.perform_test_run == True:
            if np.isnan(auc_value): 
                auc_value = -1

        auc_value_dict[endpoint] = round(auc_value, cfg.nr_of_decimals)

    # find the average of all the toxicities AUCs
    avg_auc_value = np.mean(list(auc_value_dict.values()))
    
    return avg_auc_value, auc_value_dict


"""
MSE METRICS
"""
def compute_mse(config, y_pred_list, y_true_list, mse_metric, nr_of_decimals=3):
    mse_value = mse_metric(y_pred_list, y_true_list).mean().item()
    return round(mse_value, config.nr_of_decimals)






class BestMetricsTracker():
    def __init__(self, config) -> None:
        self.best_val_epoch_num = 0

        self.best_val_avg_loss = np.inf
        self.best_val_loss_dict = {endpoint: np.inf for endpoint in config.endpoint_list} 

        self.best_val_avg_auc = 0
        self.best_val_auc_dict = {0 for _ in config.endpoint_list}

        self.nr_epochs_not_improved = 0
        self.nr_epochs_not_improved_dict = {endpoint: 0 for endpoint in config.endpoint_list}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
