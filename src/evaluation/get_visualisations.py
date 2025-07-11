import os
import pandas as pd

from src.visualization.calibration.adaptive_make_endpoint_plots import adaptive_make_endpoint_plots
from src.evaluation.utils.get_predictions_and_labels_from_predictions_dataframe import get_predictions_and_labels_from_predictions_dataframe
from src.utils.saving.get_predictions_csv_dir import get_predictions_csv_dir
from src.utils.saving.alter_filename_for_external_dataset import alter_filename_if_external_dataset

from src.constants import METRIC_TYPES, METRICS_PER_ENDPOINT_TYPE



def get_visualizations(config, sets=['train', 'val'], pred_csv_dir=None, external_set=False, is_test_set=False, lr: bool = False):
    """
    A function that creates all of the desired plots, using the predictions csv.
    Args:
        config (dict): 
        sets (list): names of the sets to make plots for
        pred_csv_dir: override for the directory to use to load the predictions csv
        external_set (bool) : if this is a external set
        is_test_set (bool): if this is a test set
    Returns:
        None

    """

    # Load the predictions csv file
    if pred_csv_dir is not None:
        predictions_csv_dir = pred_csv_dir # override the directory in the config
    else:
        # gets the directory from the config
        predictions_csv_dir = get_predictions_csv_dir(config, test_set=is_test_set, ensemble_predictions=external_set)

    df_fold_all_preds = pd.read_csv(predictions_csv_dir, sep=";")


    endpoint_list = config['columns']['labels']
    n_bins = config['evaluation']['visualisations']['n_bins']
    visualisations_list = config['evaluation']['visualisations']['list']

    # loops over each dataset
    for set_name in sets:
        # load (and drop missing) predictions and labels for each endpoint on this set
        predictions_per_endpoint_dict, labels_per_endpoint_dict = get_predictions_and_labels_from_predictions_dataframe(config, df_fold_all_preds, set_name)

        # loop over the endpoint types
        for endpoint_type, metric_types_list in METRICS_PER_ENDPOINT_TYPE.items():

            # get only the endpoints of the current endpoint_type (i.e. all Binary endpoints, or all Event endpoints)
            plotting_endpoints = [e for (e, t) in zip(config['columns']['labels'], config['columns']['labels_types']) if t == endpoint_type]
            
            # if there are no endpoints of this type, skip
            if len(plotting_endpoints) == 0:
                continue

            relevant_labels_per_endpoint_dict = {k: v for k, v in labels_per_endpoint_dict.items() if k in plotting_endpoints}
            relevant_predictions_per_endpoint_dict = {k: v for k, v in predictions_per_endpoint_dict.items() if k in plotting_endpoints}


            # init a plotting dict
            plotting_dict = [{
                "name" : set_name,
                "labels" : relevant_labels_per_endpoint_dict,
                "preds" : relevant_predictions_per_endpoint_dict
            }]
            

            # LOOP THROUGH THE PLOTS

            if endpoint_type == 'Binary':
                # calibration plot
                if 'calibration' in visualisations_list:
                    if not lr:  
                        filename = f"calibration_plot_{set_name}.png"
                    else:
                        filename = f"calibration_plot_LR_{set_name}.png"
                    filename = alter_filename_if_external_dataset(config, filename)
                    save_dir = os.path.join(config['general']['resultsCurrentDirectory'], filename)
                    adaptive_make_endpoint_plots(config, plotting_dict, column_names=plotting_endpoints, mode="calibration", title=f"Calibration Plot: {set_name} set", filedir=save_dir, return_fig=False)

                # reliability plot
                if 'reliability' in visualisations_list:
                    if not lr:  
                        filename = f"reliability_plot_{set_name}.png"
                    else:
                        filename = f"reliability_plot_LR_{set_name}.png"
                    filename = alter_filename_if_external_dataset(config, filename)
                    save_dir = os.path.join(config['general']['resultsCurrentDirectory'], filename)
                    adaptive_make_endpoint_plots(config, plotting_dict, column_names=plotting_endpoints, mode="reliability", title=f"Reliability Plot: {set_name} set", filedir=save_dir, return_fig=False)

                # confusion matrix
                if 'confusion_matrix' in visualisations_list:  
                    if not lr: 
                        filename = f"confusion_matrix_plot_{set_name}.png"
                    else:
                        filename = f"confusion_matrix_plot_LR_{set_name}.png"
                    filename = alter_filename_if_external_dataset(config, filename)
                    save_dir = os.path.join(config['general']['resultsCurrentDirectory'], filename)
                    adaptive_make_endpoint_plots(config, plotting_dict, column_names=plotting_endpoints, mode="confusion_matrix", title=f"Confusion Matrices: {set_name} set", filedir=save_dir, return_fig=False)  

                # ROC curve
                if 'roc_curve' in visualisations_list:  
                    if not lr:      
                        filename = f"ROC_curve_plot_{set_name}.png"
                    else:
                        filename = f"ROC_curve_plot_LR_{set_name}.png"
                    filename = alter_filename_if_external_dataset(config, filename)
                    save_dir = os.path.join(config['general']['resultsCurrentDirectory'], filename)
                    adaptive_make_endpoint_plots(config, plotting_dict, column_names=plotting_endpoints, mode="roc_curve", title=f"ROC curves: {set_name} set", filedir=save_dir, return_fig=False)  

            elif endpoint_type == 'Event':
                if 'kaplan_meier' in visualisations_list:
                    if not lr:  
                        filename = f"Kaplan_Meier_plot_{set_name}.png"
                    else:
                        filename = f"Kaplan_Meier_plot_LR_{set_name}.png"
                    filename = alter_filename_if_external_dataset(config, filename)
                    save_dir = os.path.join(config['general']['resultsCurrentDirectory'], filename)
                    adaptive_make_endpoint_plots(config, plotting_dict, column_names=plotting_endpoints, mode="kaplan_meier", title=f"Kaplan-Meier Plots: {set_name} set", filedir=save_dir, return_fig=False)
                    
        
            else:
                pass
    
    return

