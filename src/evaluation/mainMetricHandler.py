import numpy as np

from src.evaluation.calculate_metric_for_multiple_endpoints import calculate_metric_for_multiple_endpoints
from src.evaluation.get_metric_function import get_metric_function

class mainMetricHandler():
    """
    A class to handle the main metric calculation for the evaluation of the model.
    This object is passed around in the train() and validate() functions in the main training loop, and is used for the per-epoch evaluation of the model.
    """

    def __init__(self, config, LabelTypesManager=None):
        self.config = config

        #self.metric_name = self.config['evaluation']['main_metric']['binary_endpoints']

        self.set_metric_functions()

        self.LabelTypesManager = LabelTypesManager

    def set_metric_functions(self):

        self.metric_functions = {}
        self.metric_names_list = []

        for endpoint_type, metric_name in self.config['evaluation']['main_metric'].items():
            if endpoint_type in self.config['columns']['labels_types']:

                self.metric_functions[endpoint_type] = get_metric_function(metric_name)

        # also make a list of metric names for the main metric (for logging and printing purposes)
        for endpoint_type in self.config['columns']['labels_types']:
            self.metric_names_list.append(self.config['evaluation']['main_metric'][endpoint_type])

    def calculate_metric(self, y_pred_list_dict: dict, y_true_list_dict: dict):
        # compute the metric value for each endpoint
        mean_metric_value, results_dict = calculate_metric_for_multiple_endpoints(self.config, y_pred_list_dict, y_true_list_dict, self.metric_functions)
    
        return mean_metric_value, results_dict
    
    def calculate_mixup_metric(self, y_pred_list_dict: dict, y_true_list_dict: dict, mixup_lambda_list, mixup_indices_list):
        # function to compute the metric values if MixUp augmentation is applied to the training set
        mixup_indices_list = mixup_indices_list.cpu().tolist()
        mixup_lambda_list = mixup_lambda_list.cpu().tolist()

        y_true_list_dict_A = y_true_list_dict
        y_true_list_dict_B = {k: [v[i] for i in mixup_indices_list] for k, v in y_true_list_dict.items()}

        y_true = {k : np.concatenate([y_true_list_dict_A[k], y_true_list_dict_B[k]]) for k in y_true_list_dict_A.keys()}
        y_pred = {k : np.concatenate([y_pred_list_dict[k], y_pred_list_dict[k]]) for k in y_pred_list_dict.keys()}

        weights = np.concatenate([mixup_lambda_list, [1 - x for x in  mixup_lambda_list]])

        mean_metric_value, results_dict = calculate_metric_for_multiple_endpoints(self.config, y_pred, y_true, self.metric_functions, sample_weights=weights)

        return mean_metric_value, results_dict
