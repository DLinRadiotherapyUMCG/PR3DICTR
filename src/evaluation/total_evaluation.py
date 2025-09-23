import pandas as pd
import os
import numpy as np
import logging

from src.evaluation.get_metric_function import get_metric_function
from src.constants import METRIC_TYPES, METRICS_PER_ENDPOINT_TYPE
from src.evaluation.calculate_metric_for_multiple_endpoints import calculate_metric_for_multiple_endpoints
from src.evaluation.utils.get_predictions_and_labels_from_predictions_dataframe import get_predictions_and_labels_from_predictions_dataframe
from src.utils.saving.get_predictions_csv_dir import get_predictions_csv_dir
from src.utils.saving.alter_filename_for_external_dataset import alter_filename_if_external_dataset

def total_evaluation_current_fold(config: dict, sets: list = ['train', 'val'], is_test_set : bool = False, external_set: bool = False, pred_csv_dir: bool = None, lr: bool = False):
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
        predictions_csv_dir = get_predictions_csv_dir(config, test_set=is_test_set, ensemble_predictions=external_set)

    df_fold_all_preds = pd.read_csv(predictions_csv_dir, sep=";")

    # loop over the train, val or test sets
    for set_name in sets:
        combined_metrics_df = None

        # Replace the placeholder with a call to the new function
        predictions_per_endpoint_dict, labels_per_endpoint_dict = get_predictions_and_labels_from_predictions_dataframe(config, df_fold_all_preds, set_name)
        metrics_to_calculate = config['evaluation']['metrics_list']

        # loop over the endpoint types
        for endpoint_type, metric_types_list in METRICS_PER_ENDPOINT_TYPE.items():

            # get only the endpoints of the current endpoint_type (i.e. all Binary endpoints, or all Event endpoints)
            relevant_endpoints = [e for (e, t) in zip(config['columns']['labels'], config['columns']['labels_types']) if t == endpoint_type]

            predictions_dict_for_endpoint_type = {k: v for k, v in predictions_per_endpoint_dict.items() if k in relevant_endpoints}
            labels_dict_for_endpoint_type = {k: v for k, v in labels_per_endpoint_dict.items() if k in relevant_endpoints}

            # loop over the different metrics for this endpoint type
            for metric_type in metric_types_list:
                for metric_name in METRIC_TYPES[metric_type]:
                    
                    if metric_name in metrics_to_calculate:
                                
                        # get the metric function
                        metric_function_dict = {endpoint_type : get_metric_function(metric_name)} 

                        # calculate the metric for each endpoint
                        mean_metric_value, results_dict = calculate_metric_for_multiple_endpoints(config, predictions_dict_for_endpoint_type, labels_dict_for_endpoint_type, metric_function_dict)

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

        if lr:
            filename = f"LR_{set_name}_metrics.csv"
        else:
            filename = f"{set_name}_metrics.csv"

        filename = alter_filename_if_external_dataset(config, filename)
        combined_metrics_csv_dir = os.path.join(config['general']['resultsCurrentDirectory'], filename)
        combined_metrics_df.to_csv(combined_metrics_csv_dir, sep=";")

    logging.info(f"Total evaluation for current fold completed. Results saved in {config['general']['resultsCurrentDirectory']}")