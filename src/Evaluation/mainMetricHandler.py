


from src.evaluation.per_endpoint_metric import calculate_metric_for_multiple_endpoints
from src.evaluation.metrics.classification import auc, auc_se, balanced_acc, accuracy, F1_score, precision, recall
from src.evaluation.metrics.regression import r2, MSE, MAE, RMSE
from src.evaluation.metrics.calibration import brier, ECE, MCE, ACE
###

class mainMetricHandler():

    def __init__(self, config):
        self.config = config
        self.metric_name = self.config['evaluation']['main_metric']
        self.metric_function = self.get_metric_function()

    def get_metric_function(self):
        # Classification metrics
        if self.metric_name == 'AUC':
            return auc # self.compute_auc(predictions, labels)
        elif self.metric_name == 'AUC_SE':
            return auc_se
        elif self.metric_name == 'accuracy':
            return accuracy
        elif self.metric_name == 'balanced_accuracy':
            return balanced_acc
        elif self.metric_name == 'F1_score':
            return F1_score
        elif self.metric_name == 'precision':
            return precision
        elif self.metric_name == 'recall':
            return recall
        
        # Regression metrics
        elif self.metric_name == 'r2':
            return r2
        elif self.metric_name == 'MSE':
            return MSE
        elif self.metric_name == 'MAE':
            return MAE
        elif self.metric_name == 'RMSE':
            return RMSE

        # Calibration metrics
        elif self.metric_name == 'brier':
            return brier
        elif self.metric_name == 'ECE':
            return ECE
        elif self.metric_name == 'MCE':
            return MCE
        elif self.metric_name == 'ACE':
            return ACE

        else:
            raise ValueError(f"Unsupported metric: {self.metric_name}")

    def calculate_metric(self, y_pred_list_dict: dict, y_true_list_dict: dict):
        mean_metric_value, results_dict = calculate_metric_for_multiple_endpoints(self.config, y_pred_list_dict, y_true_list_dict, self.metric_function)
    
        return mean_metric_value, results_dict