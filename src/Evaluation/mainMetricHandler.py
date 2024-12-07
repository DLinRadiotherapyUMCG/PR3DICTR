


from src.evaluation.per_endpoint_metric import calculate_metric_for_multiple_endpoints
from src.evaluation.metrics.AUC import calculate_auc
###

class mainMetricHandler():

    def __init__(self, config):
        self.config = config
        self.metric_name = self.config['evaluation']['main_metric']
        self.metric_function = self.get_metric_function()

    def get_metric_function(self):
        if self.metric_name == 'AUC':
            return calculate_auc # self.compute_auc(predictions, labels)
        elif self.metric_name == 'MSE':
            return None # 
        else:
            raise ValueError(f"Unsupported metric: {self.metric_name}")

    def calculate_metric(self, y_pred_list_dict, y_true_list_dict):
        mean_metric_value, results_dict = calculate_metric_for_multiple_endpoints(self.config, y_pred_list_dict, y_true_list_dict, self.metric_function)
    
        return mean_metric_value, results_dict