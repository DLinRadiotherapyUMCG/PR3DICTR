import pandas as pd
import os
import numpy as np
import logging

from src.evaluation.get_evaluation_metric import get_metric_function
from src.constants import METRIC_TYPES
from src.evaluation.per_endpoint_metrics import calculate_metric_for_multiple_endpoints
from src.evaluation.utils.get_predictions_and_labels_from_predictions_dataframe import get_predictions_and_labels_from_predictions_dataframe
from src.utils.saving.get_predictions_csv_dir import get_predictions_csv_dir
from src.utils.saving.alter_filename_for_external_dataset import alter_filename_if_external_dataset


"""
1. List of metrics in config
2. List of metric_types in constants.py
3. loop over predictions.csv files in the trial's fold folders
4. compute metric per endpoint
5. mean metric value across all endpoints
6. store results in a dataframe

"""



def total_evaluation_current_fold(config: dict, sets: list = ['train', 'val'], is_test_set : bool = False, external_set: bool = False, pred_csv_dir: bool = None):
    """
    A function to calculate the evaluation metrics for the current fold. The current directory is assumed to be the fold directory in the config, but can be manually overwritten.
    Metrics are calculated for each endpoint, and each set, and the results are stored in a csv file (one file per set).
    
    Args:
        config (dict): config dictionary
        sets (list): list of strings, containing the names of the sets to evaluate (e.g. ['train', 'val', 'test'])
        external_set (bool): whether the set is an external set or not
        pred_csv_dir (str): the directory of the predictions csv file
    Returns
        None
    """
    
    logging.info(f"Starting total evaluation for current fold. Sets: {sets}")

    # Load the predictions csv file
    if pred_csv_dir is not None:
        predictions_csv_dir = pred_csv_dir
    else:
        # if external_set:
        #     #print(model_name)
        #     predictions_csv_dir = os.path.join(config['general']['resultsCurrentDirectory'], "model_predictions_external.csv")
        # elif is_test_set:
        #     predictions_csv_dir = os.path.join(config['general']['resultsCurrentDirectory'], "model_predictions_test.csv")
        # else:
        #     predictions_csv_dir = os.path.join(config['general']['resultsCurrentDirectory'], "model_predictions.csv")
        predictions_csv_dir = get_predictions_csv_dir(config, test_set=is_test_set, ensemble_predictions=external_set)

    df_fold_all_preds = pd.read_csv(predictions_csv_dir, sep=";")


    # loop over the train, val or test sets
    for set_name in sets:
        combined_metrics_df = None

        # Replace the placeholder with a call to the new function
        predictions_per_endpoint_dict, labels_per_endpoint_dict = get_predictions_and_labels_from_predictions_dataframe(config, df_fold_all_preds, set_name)

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
        combined_metrics_df = combined_metrics_df.sort_index()

        filename = f"all_{set_name}_metrics.csv"
        filename = alter_filename_if_external_dataset(config, filename)
        combined_metrics_csv_dir = os.path.join(config['general']['resultsCurrentDirectory'], filename)
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
    

