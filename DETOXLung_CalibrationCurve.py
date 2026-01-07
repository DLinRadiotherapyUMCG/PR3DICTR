import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Set working directory to --> "Pred_RT"
import sys
from pathlib import Path
path_src = os.getcwd()
sys.path.insert(1, path_src)

from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset
from src.models.tools.save_model import save_model
from src.training.train import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.hyper_opt.OptunaExperimentManager import OptunaExperimentManager
from src.utils.fileHandler import create_file, create_textfile
from src.config_presets.tools.load_config import load_config
import src.visualization.calibration.temperature_scaling

from src.evaluation.validate_on_test_set import validate_models_on_test_set
import src.visualization.calibration.temperature_scaling as ts
from src.visualization.calibration.adaptive_make_endpoint_plots import adaptive_make_endpoint_plots

from src.utils.saving.get_predictions_csv_dir import get_predictions_csv_dir
from src.evaluation.utils.get_predictions_and_labels_from_predictions_dataframe import get_predictions_and_labels_from_predictions_dataframe


def get_and_filter_preds(preds, labels):
    #mask = labels.isin([0, 1])

    #preds = preds[mask]
    #labels = labels[mask]

    preds = torch.tensor(preds)
    labels = torch.tensor(labels)

    preds_logits = torch.log(preds / (1 - preds))
    return preds_logits, preds, labels



if __name__ == '__main__':
    # Setup
    log_level = parse_args()
    setup_logging(log_level)

    # Load config for certain trial
    trial_dir = r"C:\Users\WalR\OneDrive - UMCG\Downloads\Habrok\Habrok_ResNet18-02_2025\Trial_81"
    trial_firstFold_dir= os.path.join(trial_dir, "KFold1")
    configlocation = os.path.join(trial_firstFold_dir,"DlModel_Config.yaml")
    print(configlocation)
    config = load_config(configlocation)

    #config['paths']['csv'] = r"C:/Users/r.van.der.wal/Documents/Data/2024_09_04/ClinicalData/2024_11_01/"
    #config['paths']['images'] = r"C:/Users/r.van.der.wal/Documents/Data/2024_09_04/ImageData/"
    config['training']['batch_size'] = 1

    # Load the environment
    # The load directory contains subdirectories of the folds with the given: model weights, patient cohort, etc.
    result_directory = os.path.join(trial_dir,"CalibrationCurve")

    nameModel = "ResNet10"
    model_weights = "DlModel_Weights.pth"

    prediction_scaling_methods = {
        "temperature scaling" : ts.TemperatureScaler() , 
        "platt scaling" : ts.PlattScaler(), 
        "isotonic regression": ts.IsotonicRegressionScaler(), 
        "beta calibration": ts.BetaCalibration()
        }

    original_results = {}
    method_results = {}

    # Folds is dependent on amount of directories
    #folders = os.listdir(trial_dir)
    folders = [name for name in os.listdir(trial_dir) if os.path.isdir(os.path.join(trial_dir, name))]
    foldAmount = len(folders)

    # Load model paths
    path_csv = config['paths']['csv']#r"C:/Users/r.van.der.wal/Documents/Data/2024_09_04/ClinicalData/2024_11_01/"
    path_images = config['paths']['images']# r"C:/Users/r.van.der.wal/Documents/Data/2024_09_04/ImageData/"

    for scaling_method, posthoc_model in prediction_scaling_methods.items():

        endpointArray = config['columns']['labels']
        ens_preds_dict = {endpoint: None for endpoint in endpointArray}
        scaled_ens_preds_dict = {endpoint: None for endpoint in endpointArray}
        true_labels_dict = {endpoint: None for endpoint in endpointArray}

        for folder in folders:     # one folder per cross-validation fold
            fold_result_dir = os.path.join(result_directory ,folder)
            fold_dir = os.path.join(trial_dir ,folder)

            # WE ASSUME THE FOLDS ARE ALLREADY CHECKED!!!
            #if(os.path.exists(fold_result_dir) == False):
            #    fold_proc_dir = os.path.join(trial_dir, folder)
            #    total_evaluation(fold_proc_dir, fold_result_dir, path_csv, path_images, True, 1, single = True)
            #    total_evaluation(fold_proc_dir, fold_result_dir, path_csv, path_images, True, 2, single = True)
            set_name = 'val'
            predictions_csv_dir = os.path.join(fold_dir,"model_predictions.csv")
            df_fold_all_preds = pd.read_csv(predictions_csv_dir, sep=";")
            df_val_preds, df_val_label = get_predictions_and_labels_from_predictions_dataframe(config, df_fold_all_preds, set_name)

            set_name = 'test'
            predictions_csv_dir = os.path.join(fold_dir,"test_set_predictions.csv")
            df_fold_all_preds = pd.read_csv(predictions_csv_dir, sep=";")
            df_test_preds, df_test_label = get_predictions_and_labels_from_predictions_dataframe(config, df_fold_all_preds, set_name)

            #for endpoint in config.endpoint_list:
            # first find the scaling by using the validation set (train the posthoc model)
            endpoint = config['columns']['labels'][0]
            #print(df_val_preds)

            val_logits, val_preds, val_labels = get_and_filter_preds((df_val_preds[endpoint]), (df_val_label[endpoint]))
            posthoc_model.fit(val_logits, val_labels)

            # use the posthoc model to scale the test set predictions
            test_logits, test_preds, test_labels = get_and_filter_preds((df_test_preds[endpoint]), (df_test_label[endpoint]))
            test_logreg_preds = posthoc_model.predict_proba(test_logits)
            test_logreg_preds = test_logreg_preds

            # save the (ensemble) predictions
            if ens_preds_dict[endpoint] is None:
                ens_preds_dict[endpoint] = test_preds/foldAmount
                scaled_ens_preds_dict[endpoint] = test_logreg_preds/foldAmount
                true_labels_dict[endpoint] = test_labels
            else:
                ens_preds_dict[endpoint] += test_preds/foldAmount
                scaled_ens_preds_dict[endpoint] += test_logreg_preds/foldAmount
    
        method_results[scaling_method] = scaled_ens_preds_dict


    method_results["original"] = ens_preds_dict

    # Part 3: Making the plots
    nameModel = "ResNet18"
    resultDir_Plots = result_directory

    big_results_dict = [
        {
            "name": "Original",
            "preds": method_results["original"], 
            "labels": true_labels_dict
        }, {
            "name": "Temperature", 
            "preds": method_results["temperature scaling"], 
            "labels": true_labels_dict
        }, 
        {
            "name": "Platt", 
            "preds": method_results["platt scaling"], 
            "labels": true_labels_dict
        },
        {"name": "Isotonic", "preds": method_results["isotonic regression"], "labels": true_labels_dict},
        {"name": "Beta", "preds": method_results["beta calibration"], "labels": true_labels_dict}
    ]

    rel_plot_export_dir = os.path.join(resultDir_Plots,"Temperature Scaling", f"{nameModel}_reliability.png")
    cal_plot_export_dir = os.path.join(resultDir_Plots,"Temperature Scaling", f"{nameModel}_calibration.png")

    endpointArray = config['columns']['labels']
    #fig = adaptive_make_calibration_plots(config, row_dicts=big_results_dict, column_names=endpointArray, title=nameModel, 
    #                                    mode='reliability', filedir=rel_plot_export_dir, return_fig=True)

    print("Creating figure")
    adaptive_make_endpoint_plots(config, row_dicts=big_results_dict, column_names=endpointArray, title=nameModel, 
                                        mode='calibration', filedir=cal_plot_export_dir, return_fig=False)
