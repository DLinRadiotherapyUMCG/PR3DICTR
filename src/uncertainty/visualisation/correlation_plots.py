import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_curve, roc_auc_score, accuracy_score
from lifelines.utils import concordance_index

from src.uncertainty.visualisation.calibration_plot import normalise_uncertainty_values

from scipy.stats import spearmanr, pearsonr

def plot_uncertainty_and_error_correlation_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, colours_dict, normalisation_method="minmax"):
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray') # , label='Ideal Calibration')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')

    metric_name_map = {"Binary Entropy": "BE",
                       "Variance": "V",
                       "Mutual Information": "MI",}

    if ENDPOINT_TYPES[endpoint] == "Binary":
        df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Labels'] != -1]
        true_labels = df_UQ_no_missing_labels['True Labels'].values
    else:
        df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Label Event'] != -1]
        true_labels = df_UQ_no_missing_labels[['True Label Event', 'True Months Event']].values

    mean_preds = df_UQ_no_missing_labels['Mean Prediction'].values

    for UQ_metric_name in UQ_metrics_list:
        UQ_metric = df_UQ_no_missing_labels[UQ_metric_name].values
        if UQ_metric_name == "Binary Entropy":
            UQ_metric_norm = UQ_metric
        else:
            UQ_metric_norm = normalise_uncertainty_values(UQ_metric, normalisation_method=normalisation_method)

        # calculate the individual brier components
        if ENDPOINT_TYPES[endpoint] == "Binary":
            errors = (true_labels - mean_preds)**2
        else:
            errors = (true_labels[:,0] - mean_preds)**2  # using event indicator as a proxy for error

        # Print Spearman correlation between uncertainty and errors
        corr, pval = spearmanr(UQ_metric_norm, errors)
        # print(f"Spearman correlation between {endpoint} {metric_name_map[UQ_metric_name]} and errors: {corr:.3f} (p={pval:.3g})")

        
        ax.scatter(UQ_metric_norm, errors, label=f"{metric_name_map[UQ_metric_name]} Spearman={corr:.2f}", color=colours_dict[UQ_metric_name], alpha=0.6, s=10)


    ax.legend(loc='lower right', fontsize='small')        


    #ax.set_xlabel("Uncertainty")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)









def plot_uncertainty_and_error_AUROC_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, colours_dict, normalisation_method="minmax"):
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray') # , label='Ideal Calibration')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')

    metric_name_map = {"Binary Entropy": "BE",
                       "Variance": "V",
                       "Mutual Information": "MI",}

    if ENDPOINT_TYPES[endpoint] == "Binary":
        df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Labels'] != -1]
        true_labels = df_UQ_no_missing_labels['True Labels'].values
    else:
        df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Label Event'] != -1]
        true_labels = df_UQ_no_missing_labels[['True Label Event', 'True Months Event']].values

    mean_preds = df_UQ_no_missing_labels['Mean Prediction'].values

    for UQ_metric_name in UQ_metrics_list:
        UQ_metric = df_UQ_no_missing_labels[UQ_metric_name].values
        if UQ_metric_name == "Binary Entropy":
            UQ_metric_norm = UQ_metric
        else:
            UQ_metric_norm = normalise_uncertainty_values(UQ_metric, normalisation_method=normalisation_method)

        # calculate the individual brier components
        if ENDPOINT_TYPES[endpoint] == "Binary":
            errors = (mean_preds - true_labels)**2
        else:
            errors = (mean_preds - true_labels[:,0])**2  # using event indicator as a proxy for error



        # Compute AUROC for error | uncertainty
        # Binarize errors: 1 if error > median, else 0
        error_binary = (errors > np.mean(errors)).astype(int)
        auc = roc_auc_score(error_binary, UQ_metric_norm)
        fpr, tpr, _ = roc_curve(error_binary, UQ_metric_norm)
        ax.plot(fpr, tpr, label=f"{metric_name_map[UQ_metric_name]} AUC={auc:.2f}", color=colours_dict[UQ_metric_name], linewidth=2)

        # visually check if mean or median threshold makes a difference
        # ax.hist(errors, bins=50)
        # ax.axvline(np.mean(errors), color='red', linestyle='--', label='Mean Error')
        # ax.axvline(np.median(errors), color='blue', linestyle='--', label='Median Error')



        # plot a dashed line AUC curve for binary correctness
        # use binary error
        # error_binary = (mean_preds > 0.5) != true_labels.astype(int)
        # error_binary = error_binary.astype(float)
        # auc_correctness = roc_auc_score(error_binary, UQ_metric_norm)
        # fpr_c, tpr_c, _ = roc_curve(error_binary, UQ_metric_norm)
        # ax.plot(fpr_c, tpr_c,  label=f"{metric_name_map[UQ_metric_name]} C AUC={auc_correctness:.2f}", color=colours_dict[UQ_metric_name], linewidth=2)


    ax.legend(loc='lower right', fontsize='small')        


    #ax.set_xlabel("Uncertainty")
    ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1.0)

