import numpy as np


from src.evaluation.per_endpoint_metrics import calculate_metric_for_multiple_endpoints
from src.evaluation.get_evaluation_metric import get_metric_function

###

class mainMetricHandler():
    """
    A class to handle the main metric calculation for the evaluation of the model.
    This object is passed around in the train() and validate() functions in the main training loop, and is used for the per-epoch evaluation of the model.
    """

    def __init__(self, config):
        self.config = config
        self.metric_name = self.config['evaluation']['main_metric']
        self.metric_function = get_metric_function(self.metric_name)

    def calculate_metric(self, y_pred_list_dict: dict, y_true_list_dict: dict):
        mean_metric_value, results_dict = calculate_metric_for_multiple_endpoints(self.config, y_pred_list_dict, y_true_list_dict, self.metric_function)
    
        return mean_metric_value, results_dict
    
    def calculate_mixup_metric(self, y_pred_list_dict: dict, y_true_list_dict: dict, mixup_lambda_list, mixup_indices_list):

        mixup_indices_list = mixup_indices_list.cpu().tolist()
        mixup_lambda_list = mixup_lambda_list.cpu().tolist()


        y_true_list_dict_A = y_true_list_dict
        y_true_list_dict_B = {k: [v[i] for i in mixup_indices_list] for k, v in y_true_list_dict.items()}

        #y_true_list_dict_A = {k : [v_i / lmb_i for v_i, lmb_i in zip(v, mixup_lambda_list)] for k, v in y_true_list_dict_A.items()}
        #y_true_list_dict_B = {k : [v_i / (1-lmb_i) for v_i, lmb_i in zip(v, mixup_lambda_list)] for k, v in y_true_list_dict_B.items()}

        y_true = {k : np.concatenate([y_true_list_dict_A[k], y_true_list_dict_B[k]]) for k in y_true_list_dict_A.keys()}
        y_pred = {k : np.concatenate([y_pred_list_dict[k], y_pred_list_dict[k]]) for k in y_pred_list_dict.keys()}

        weights = np.concatenate([mixup_lambda_list, [1 - x for x in  mixup_lambda_list]])


        mean_metric_value, results_dict = calculate_metric_for_multiple_endpoints(self.config, y_pred, y_true, self.metric_function, sample_weights=weights)
        
        #
        
        # mean_metric_value_A, results_dict_A = calculate_metric_for_multiple_endpoints(self.config, y_pred_list_dict, y_true_list_dict_A, self.metric_function, sample_weights=mixup_lambda_list.cpu().tolist())
        # mean_metric_value_B, results_dict_B = calculate_metric_for_multiple_endpoints(self.config, y_pred_list_dict, y_true_list_dict_B, self.metric_function, sample_weights=(1 - mixup_lambda_list).cpu().tolist())

        # mean_metric_value = (mean_metric_value_A + mean_metric_value_B ) / 2
        # results_dict = {k: (v + results_dict_B[k] )/2 for k, v in results_dict_A.items()}

        #mean_metric_value = (mean_metric_value_A * mixup_lambda_list + mean_metric_value_B * (1 - mixup_lambda_list))
        #results_dict = {k: (v * mixup_lambda_list + results_dict_B[k] * (1 - mixup_lambda_list)) for k, v in results_dict_A.items()}


        return mean_metric_value, results_dict


