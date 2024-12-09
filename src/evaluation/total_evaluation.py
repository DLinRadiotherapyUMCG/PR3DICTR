import pandas as pd
import os
import numpy as np
import logging

from src.evaluation.get_evaluation_metric import get_metric_function
from src.constants import METRIC_TYPES
from src.evaluation.metrics.utils import remove_missing
from src.evaluation.per_endpoint_metrics import calculate_metric_for_multiple_endpoints



"""
1. List of metrics in config
2. List of metric_types in constants.py
3. loop over predictions.csv files in the trial's fold folders
4. compute metric per endpoint
5. mean metric value across all endpoints
6. store results in a dataframe

"""



def total_evaluation_current_fold(config, sets = ['train', 'val'], external_set = False, pred_csv_dir = None):
    """
    A function to calculate the evaluation metrics for the current fold. The current directory is assumed to be the fold directory in the config, but can be manually overwritten.
    """
    
    logging.info(f"Starting total evaluation for current fold. Sets: {sets}")

    # Load the predictions csv file
    if pred_csv_dir is not None:
        predictions_csv_dir = pred_csv_dir
    else:
        if external_set:
            #print(model_name)
            predictions_csv_dir = os.path.join(config['general']['resultsCurrentDirectory'], "model_predictions_external.csv")
        else:
            predictions_csv_dir = os.path.join(config['general']['resultsCurrentDirectory'], "model_predictions.csv")

    df_fold_all_preds = pd.read_csv(predictions_csv_dir, sep=";")


    # loop over the train, val or test sets
    for set_name in sets:
        combined_metrics_df = None

        # ger predictions of this set
        df_fold_preds = df_fold_all_preds[df_fold_all_preds["Mode"]==set_name]
        
        # collect all (valid) predictions and labels for each endpoint (i.e. mask missing values)
        endpoint_list = config['columns']['labels']
        predictions_per_endpoint_dict = {endpoint: None for endpoint in endpoint_list}
        labels_per_endpoint_dict = {endpoint: None for endpoint in endpoint_list}

        for endpoint in endpoint_list:
            pred = df_fold_preds[[f"{endpoint}_pred"]].to_numpy()
            true = df_fold_preds[[f"{endpoint}_true"]].to_numpy()

            # mask out the missing values
            true, pred = remove_missing(true, pred)

            # store in dict
            predictions_per_endpoint_dict[endpoint] = pred
            labels_per_endpoint_dict[endpoint] = true
        

        metrics_to_calculate = config['evaluation']['metrics_list']

        # loop over the metric types
        for metric_type, metric_names in METRIC_TYPES.items():
            for metric_name in list(metric_names):
                if metric_name in metrics_to_calculate:
                            
                    # get the metric function
                    metric_function = get_metric_function(metric_name)

                    # calculate the metric for each endpoint
                    mean_metric_value, results_dict = calculate_metric_for_multiple_endpoints(config, predictions_per_endpoint_dict, labels_per_endpoint_dict, metric_function)

                    # store the results in a dataframe
                    results_df = pd.DataFrame(results_dict, index=[metric_name])
                    results_df.insert(0, "MEAN", mean_metric_value)  # put the mean value in the first column of the dataframe

                    # append the results to a combined dataframe
                    if 'combined_results_df' == None:
                        combined_metrics_df = results_df
                    else:
                        combined_metrics_df = pd.concat([combined_metrics_df, results_df])

        # save the combined results to a csv
        combined_metrics_csv_dir = os.path.join(config['general']['resultsCurrentDirectory'], f"all_{set_name}_metrics.csv")
        combined_metrics_df.to_csv(combined_metrics_csv_dir, sep=";")








# # # FOR TESTING THIS FUNCTION (DANIEL)
# 
# if __name__ == "__main__":
#     from src.config_presets.tools.get_config import get_config


#     # Load the configuration file
#     config = get_config('Multi_tox')

#     config['general']['resultsCurrentDirectory'] = "C:/Users/S.P.M. de Vette/OneDrive - UMCG/Desktop/pred_RT_results/Results_0\Trial_2\KFold1"
#     # Call the function
#     total_evaluation_current_fold(config, external_set=False)
    

