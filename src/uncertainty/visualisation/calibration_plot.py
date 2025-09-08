import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd

from src.evaluation.metrics.classification import accuracy
from sklearn.metrics import log_loss

from sklearn.metrics import roc_curve


def normalise_uncertainty_values(UQ_metric, normalisation_method="minmax" ):
    """
    A helper function to normalise uncertainty values for plotting (reliability or calibration plots).
    Args:
        UQ_metric (np.array): array of uncertainty values
        normalisation_method (str): "minmax" or "percentile"
    Returns:
        UQ_metric_normalised (np.array): normalised uncertainty values
    """
    if normalisation_method == "minmax":
        UQ_metric_normalised = (UQ_metric - np.min(UQ_metric)) / (np.max(UQ_metric) - np.min(UQ_metric) + 1e-8)
    elif normalisation_method == "percentile":
        lower = np.percentile(UQ_metric, 5)
        upper = np.percentile(UQ_metric, 95)
        UQ_metric_normalised = (UQ_metric - lower) / (upper - lower + 1e-8)
        UQ_metric_normalised = np.clip(UQ_metric_normalised, 0, 1)
    else:
        raise ValueError(f"Unknown normalisation method: {normalisation_method}")

    return UQ_metric_normalised




def plot_calibration_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, N_bins, colours_dict, normalisation_method="minmax"):
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Calibration')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')

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

        uq_bins = pd.qcut(UQ_metric_norm, N_bins, labels=False, duplicates='drop')
        aucs, bin_centers = [], []
        for b in np.unique(uq_bins):
            idx = uq_bins == b
            if ENDPOINT_TYPES[endpoint] == "Binary":
                if np.unique(true_labels[idx]).size == 2:
                    # auc = roc_auc_score(true_labels[idx], mean_preds[idx])

                    # auc = np.mean((true_labels[idx] - mean_preds[idx]) ** 2)  # MSE
                    #config_temp = {'evaluation': {'metrics': {'threshold': 'YoudenJ'}}}
                    # from sklearn.metrics import  accuracy_score
                    #s
                    # auc = accuracy(config_temp, true_labels[idx], mean_preds[idx])
                    thresh_value = 0.5
                    auc = accuracy_score(true_labels[idx], mean_preds[idx]>thresh_value)
                    aucs.append(auc)
                else:
                    aucs.append(1)
            else:
                if np.unique(true_labels[idx][:, 0]).size == 2:
                    auc = concordance_index(true_labels[idx][:, 1], mean_preds[idx], true_labels[idx][:, 0])
                    aucs.append(auc)
                else:
                    aucs.append(1)
            bin_centers.append(1- UQ_metric_norm[idx].mean())

        ax.plot(bin_centers, aucs, 'o', label=UQ_metric_name, color=colours_dict[UQ_metric_name])
        coef = np.polyfit(bin_centers, aucs, 1)
        linear_fit = np.poly1d(coef)
        x_values = np.linspace(0, 1, 100)
        ax.plot(x_values, linear_fit(x_values), linestyle='-', color=colours_dict[UQ_metric_name], alpha=0.8, linewidth=2)

    ax.set_xlabel("Certainty (1 - uncertainty)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)





import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd

from src.evaluation.metrics.classification import accuracy
from sklearn.metrics import log_loss


def plot_error_calibration_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, N_bins, colours_dict, normalisation_method="minmax"):
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Calibration')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')

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

        uq_bins = pd.qcut(UQ_metric_norm, N_bins, labels=False, duplicates='drop')
        aucs, bin_centers = [], []
        for b in np.unique(uq_bins):
            idx = uq_bins == b
            if ENDPOINT_TYPES[endpoint] == "Binary":
                if np.unique(true_labels[idx]).size == 2:
                    bce = log_loss(true_labels[idx], mean_preds[idx])
                    #MSE = np.mean((true_labels[idx] - mean_preds[idx]) ** 2)
                    #nll = -np.mean(true_labels[idx] * np.log(mean_preds[idx] + 1e-8) + (1 - true_labels[idx]) * np.log(1 - mean_preds[idx] + 1e-8))
                    aucs.append(bce)
                else:
                    aucs.append(1)
            else:
                if np.unique(true_labels[idx][:, 0]).size == 2:
                    auc = concordance_index(true_labels[idx][:, 1], mean_preds[idx], true_labels[idx][:, 0])
                    aucs.append(auc)
                else:
                    aucs.append(1)
            bin_centers.append(UQ_metric_norm[idx].mean())

        ax.plot(bin_centers, aucs, 'o', label=UQ_metric_name, color=colours_dict[UQ_metric_name])
        coef = np.polyfit(bin_centers, aucs, 1)
        linear_fit = np.poly1d(coef)
        x_values = np.linspace(0, 1, 100)
        ax.plot(x_values, linear_fit(x_values), linestyle='-', color=colours_dict[UQ_metric_name], alpha=0.8, linewidth=2)


    ax.set_xlabel("Uncertainty")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)







def plot_calibration_error_over_dataset_size(
        ax,
    df_data_dict,
    endpoint,
    ENDPOINT_TYPES,
    UQ_metrics_list,
    N_bins,
    colours_dict,
    normalisation_method="minmax",
):
    # Prepare rows for DataFrame
    rows = []

    N_patients_list = df_data_dict.keys()
    
    for N_patients in N_patients_list:
        df_UQ_temp = df_data_dict[N_patients]

        if ENDPOINT_TYPES[endpoint] == "Binary":
            df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Labels'] != -1]
            true_labels = df_UQ_no_missing_labels['True Labels'].values
        else:
            df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Label Event'] != -1]
            true_labels = df_UQ_no_missing_labels[['True Label Event', 'True Months Event']].values

        mean_preds = df_UQ_no_missing_labels['Mean Prediction'].values
        #print(df_UQ_temp)

        for UQ_metric_name in UQ_metrics_list:

                    
            UQ_metric = df_UQ_no_missing_labels[UQ_metric_name].values
            if UQ_metric_name == "Binary Entropy":
                UQ_metric_norm = UQ_metric
            else:
                UQ_metric_norm = normalise_uncertainty_values(UQ_metric, normalisation_method=normalisation_method)

            uq_bins = pd.qcut(UQ_metric_norm, N_bins, labels=False, duplicates='drop')
            aucs, bin_centers = [], []
            for b in np.unique(uq_bins):
                idx = uq_bins == b
                if ENDPOINT_TYPES[endpoint] == "Binary":
                    if np.unique(true_labels[idx]).size == 2:
                        auc = roc_auc_score(true_labels[idx], mean_preds[idx])
                        #auc = log_loss(true_labels[idx], mean_preds[idx])
                        auc = accuracy_score(true_labels[idx], mean_preds[idx] > 0.5)
                        aucs.append(auc)
                    else:
                        aucs.append(1)
                else:
                    if np.unique(true_labels[idx][:, 0]).size == 2:
                        auc = concordance_index(true_labels[idx][:, 1], mean_preds[idx], true_labels[idx][:, 0])
                        aucs.append(auc)
                    else:
                        aucs.append(1)
                bin_centers.append(UQ_metric_norm[idx].mean())
            
            value = np.sum(abs(np.array(aucs) - np.array(bin_centers))**2) / len(aucs)
            # COMPUTE METRIC HERE
            #value = metric_func(config, aucs, bin_centers)
            rows.append({
                "endpoint": endpoint,
                "N_patients": N_patients,
                "uq_metric": UQ_metric_name,
                "metric_value": value
            })
 
    df = pd.DataFrame(rows)

    for UQ_metric_name in df['uq_metric'].unique():
        subset = df[df['uq_metric'] == UQ_metric_name]
        ax.plot(subset['N_patients'], subset['metric_value'], label=UQ_metric_name, color=colours_dict[UQ_metric_name])

    ax.set_xlabel("N training patients")

    #return df