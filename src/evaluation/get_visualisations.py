# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:25:03 2024

@author: HoekL02
"""
import os
import pandas as pd

from src.visualization.calibration.calibration_plots import adaptive_make_calibration_plots
from src.evaluation.utils.get_predictions_and_labels_from_predictions_dataframe import get_predictions_and_labels_from_predictions_dataframe
from src.utils.saving.get_predictions_csv_dir import get_predictions_csv_dir
from src.utils.saving.alter_filename_for_external_dataset import alter_filename_if_external_dataset


def get_visualizations(config, sets=['train', 'val'], pred_csv_dir=None, external_set=False, is_test_set=False):
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
        #os.path.join(config['general']['resultsCurrentDirectory'], "model_predictions.csv")

    df_fold_all_preds = pd.read_csv(predictions_csv_dir, sep=";")


    endpoint_list = config['columns']['labels']
    n_bins = config['evaluation']['visualisations']['n_bins']

    visualisations_list = config['evaluation']['visualisations']['list']

    for set_name in sets:
        
        predictions_per_endpoint_dict, labels_per_endpoint_dict = get_predictions_and_labels_from_predictions_dataframe(config, df_fold_all_preds, set_name)


        plotting_dict = [{
            "name" : set_name,
            "labels" : labels_per_endpoint_dict,
            "preds" : predictions_per_endpoint_dict
        }]


        # LOOP THROUGH THE PLOTS
        # calibration plot
        if 'calibration' in visualisations_list:
            filename = f"calibration_plot_{set_name}.png"
            filename = alter_filename_if_external_dataset(config, filename)
            save_dir = os.path.join(config['general']['resultsCurrentDirectory'], filename)
            adaptive_make_calibration_plots(config, plotting_dict, column_names=endpoint_list, n_bins=n_bins, mode="calibration", title=f"Calibration Plot: {set_name} set", filedir=save_dir, return_fig=False)
    
        # reliability plot
        if 'reliability' in visualisations_list:
            filename = f"reliability_plot_{set_name}.png"
            filename = alter_filename_if_external_dataset(config, filename)
            save_dir = os.path.join(config['general']['resultsCurrentDirectory'], filename)
            adaptive_make_calibration_plots(config, plotting_dict, column_names=endpoint_list, n_bins=n_bins, mode="reliability", title=f"Reliability Plot: {set_name} set", filedir=save_dir, return_fig=False)

        # confusion matrix
        if 'confusion_matrix' in visualisations_list:  
            filename = f"confusion_matrix_plot_{set_name}.png"
            filename = alter_filename_if_external_dataset(config, filename)
            save_dir = os.path.join(config['general']['resultsCurrentDirectory'], filename)
            adaptive_make_calibration_plots(config, plotting_dict, column_names=endpoint_list, n_bins=n_bins, mode="confusion_matrix", title=f"Confusion Matrices: {set_name} set", filedir=save_dir, return_fig=False)  

        # ROC curve
        if 'roc_curve' in visualisations_list:        
            filename = f"ROC_curve_plot_{set_name}.png"
            filename = alter_filename_if_external_dataset(config, filename)
            save_dir = os.path.join(config['general']['resultsCurrentDirectory'], filename)
            adaptive_make_calibration_plots(config, plotting_dict, column_names=endpoint_list, n_bins=n_bins, mode="roc_curve", title=f"ROC curves: {set_name} set", filedir=save_dir, return_fig=False)  

    
    return



# # # FOR TESTING THIS FUNCTION (DANIEL)

# if __name__ == "__main__":
#     from src.config_presets.tools.get_config import get_config

#     # Load the configuration file
#     config = get_config('Multi_tox')

#     config['general']['resultsCurrentDirectory'] = "C:/Users/S.P.M. de Vette/OneDrive - UMCG/Desktop/pred_RT_results/Results_0\Trial_2\KFold1"
#     # Call the function
    
#     get_visualizations(config, sets=['train', 'val'], pred_csv_dir=None, external_set=False)
    

