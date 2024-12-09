


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


