import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_curve, roc_auc_score, accuracy_score
from lifelines.utils import concordance_index

from src.evaluation.metrics.classification import accuracy


def normalise_uncertainty_values(UQ_metric, normalisation_method="minmax" ):
    """
    A helper function to normalise uncertainty values for plotting (reliability or calibration plots).
    Args:
        UQ_metric (np.array): array of uncertainty values
        normalisation_method (str): "minmax" or "percentile"
    Returns:
        UQ_metric_normalised (np.array): normalised uncertainty values
    """
    if normalisation_method is None:
        UQ_metric_normalised = UQ_metric
    elif normalisation_method == "minmax":
        UQ_metric_normalised = (UQ_metric - np.min(UQ_metric)) / (np.max(UQ_metric) - np.min(UQ_metric))
    elif normalisation_method == "percentile":
        lower = np.percentile(UQ_metric, 5)
        upper = np.percentile(UQ_metric, 95)
        UQ_metric_normalised = (UQ_metric - lower) / (upper - lower + 1e-8)
        UQ_metric_normalised = np.clip(UQ_metric_normalised, 0, 1)
    else:
        raise ValueError(f"Unknown normalisation method: {normalisation_method}")

    return UQ_metric_normalised




def plot_calibration_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, N_bins, colours_dict, 
                             plot_type="UQ_calibration", normalisation_method="minmax"):
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

    N_bins = len(mean_preds) // 25


    for UQ_metric_name in UQ_metrics_list:
        UQ_metric = df_UQ_no_missing_labels[UQ_metric_name].values
        if UQ_metric_name == "Binary Entropy":
            UQ_metric_norm = UQ_metric
        else:
            UQ_metric_norm = normalise_uncertainty_values(UQ_metric, normalisation_method=normalisation_method)

        if plot_type == "UQ_calibration":
            uq_bins = pd.qcut(UQ_metric_norm, N_bins, labels=False, duplicates='drop')
        elif plot_type == "prediction_calibration":
            uq_bins = pd.qcut(mean_preds, N_bins, labels=False, duplicates='drop')
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        aucs, bin_centers = [], []


        fpr, tpr, thresholds = roc_curve(true_labels, mean_preds)
        idx = np.argmax(tpr - fpr) # use the threshold at the biggest difference between true positive rate and false positive rate
        thresh_value = thresholds[idx]
        # thresh_value = 0.5

        print(f"Threshold for binary classification: {thresh_value:.2f}. UQ_metric_name: {UQ_metric_name}, plot_type: {plot_type}")

        for b in np.unique(uq_bins):
            idx = uq_bins == b
            if ENDPOINT_TYPES[endpoint] == "Binary":
                if plot_type == "UQ_calibration":
                    if np.unique(true_labels[idx]).size == 2:
                        #thresh_value = 0.5
                        auc = accuracy_score(true_labels[idx], mean_preds[idx]>thresh_value)
                        # auc = roc_auc_score(true_labels[idx], mean_preds[idx])
                        aucs.append(auc)
                    else:
                        aucs.append(1)                  
                    
                    
                elif plot_type == "prediction_calibration":
                    aucs.append(true_labels[idx].mean())  # plot the 'observed rate' on y-axis
                
            else:
                if np.unique(true_labels[idx][:, 0]).size == 2:
                    auc = concordance_index(true_labels[idx][:, 1], mean_preds[idx], true_labels[idx][:, 0])
                    aucs.append(auc)
                else:
                    aucs.append(1)

            if plot_type == "UQ_calibration":
                bin_centers.append(1 - UQ_metric_norm[idx].mean())
            elif plot_type == "prediction_calibration":
                bin_centers.append(mean_preds[idx].mean())

        ax.plot(bin_centers, aucs, 'o', label=UQ_metric_name, color=colours_dict[UQ_metric_name], markersize=5)
        coef = np.polyfit(bin_centers, aucs, 1)
        linear_fit = np.poly1d(coef)
        x_values = np.linspace(0, 1, 100)
        ax.plot(x_values, linear_fit(x_values), linestyle='-', color=colours_dict[UQ_metric_name], alpha=0.8, linewidth=2)

        # to note the ACE values on the plot itself
        # ACE_score = np.sum(np.abs(np.array(aucs) - (np.array(bin_centers)))) / len(aucs)
        # ax.text(0.6, 0.2 - 0.1 * UQ_metrics_list.index(UQ_metric_name), f"{ACE_score:.2f}", color=colours_dict[UQ_metric_name])

    if plot_type == "UQ_calibration":
        ax.set_xlabel("Certainty (1 - uncertainty)")
    else:
        ax.set_xlabel("Prediction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)







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

        ax.plot(bin_centers, aucs, 'o', label=UQ_metric_name, color=colours_dict[UQ_metric_name], markersize=5)
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

    N_iterations_list = list(df_data_dict.keys())
    N_patients_list = list(df_data_dict[N_iterations_list[0]].keys())
    
    for iteration in N_iterations_list:
            
        for N_patients in N_patients_list:
            df_UQ_temp = df_data_dict[iteration][N_patients]


            if ENDPOINT_TYPES[endpoint] == "Binary":
                df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Labels'] != -1]
                true_labels = df_UQ_no_missing_labels['True Labels'].values
            else:
                df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Label Event'] != -1]
                true_labels = df_UQ_no_missing_labels[['True Label Event', 'True Months Event']].values

            mean_preds = df_UQ_no_missing_labels['Mean Prediction'].values

            N_bins = len(mean_preds) // 25
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
                            # auc = roc_auc_score(true_labels[idx], mean_preds[idx])
                            # auc = log_loss(true_labels[idx], mean_preds[idx])
                            # 
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
                
                value = np.sum(abs(np.array(aucs) - (1 - np.array(bin_centers)))) / len(aucs)

                # COMPUTE METRIC HERE
                #value = metric_func(config, aucs, bin_centers)
                rows.append({
                    "endpoint": endpoint,
                    "N_patients": int(N_patients),
                    "uq_metric": UQ_metric_name,
                    "metric_value": value
                })
    
    df = pd.DataFrame(rows)

    grouped = df.groupby(['N_patients', 'uq_metric'])['metric_value'].agg(['mean', 'std']).reset_index()
    for uq_metric in grouped['uq_metric'].unique():
        subset = grouped[grouped['uq_metric'] == uq_metric]# .sort_values('N_patients')
        #print(subset)
        ax.errorbar(
            subset['N_patients'],
            subset['mean'],
            yerr=subset['std'],
            label=uq_metric,
            marker='o',
            capsize=3,
            color=colours_dict[uq_metric]
        )
    

    ax.set_xlabel("N training patients")

    #return df



def plot_UQ_values_over_dataset_size(
        ax,
    df_data_dict,
    endpoint,
    ENDPOINT_TYPES,
    UQ_metrics_list,
    N_bins,
    colours_dict,
    normalisation_method="minmax",
    is_last_col = True
):
    # Prepare rows for DataFrame
    rows = []

    N_iterations_list = list(df_data_dict.keys())
    N_patients_list = list(df_data_dict[N_iterations_list[0]].keys())
    
    for iteration in N_iterations_list:
            
        for N_patients in N_patients_list:
            df_UQ_temp = df_data_dict[iteration][N_patients]


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

                mean_UQ_value = np.mean(UQ_metric_norm)
                rows.append({
                    "endpoint": endpoint,
                    "N_patients": int(N_patients),
                    "uq_metric": UQ_metric_name,
                    "mean_UQ_value": mean_UQ_value
                })

    
    df = pd.DataFrame(rows)

    grouped = df.groupby(['N_patients', 'uq_metric'])['mean_UQ_value'].agg(['mean', 'std']).reset_index()
    right_ax = None
    for uq_metric in grouped['uq_metric'].unique():
        subset = grouped[grouped['uq_metric'] == uq_metric]  # .sort_values('N_patients')
        if uq_metric == "Binary Entropy":
            # create secondary y-axis once
            if right_ax is None:
                right_ax = ax.twinx()
                right_ax.set_ylim(0, 1)
                right_ax.tick_params(axis='y', labelcolor=colours_dict[uq_metric])
                right_ax.spines['right'].set_color(colours_dict[uq_metric])
                # right_ax.set_ylabel("Binary Entropy")
            right_ax.errorbar(
                subset['N_patients'],
                subset['mean'],
                yerr=subset['std'],
                label=uq_metric,
                marker='o',
                capsize=3,
                color=colours_dict[uq_metric]
            )
            if not is_last_col:
                right_ax.yaxis.set_visible(False)
                #right_ax.spines['right'].set_visible(False)
        else:
            ax.errorbar(
                subset['N_patients'],
                subset['mean'],
                yerr=subset['std'],
                label=uq_metric,
                marker='o',
                capsize=3,
                color=colours_dict[uq_metric]
            )
    

    ax.set_xlabel("N training patients")




def plot_accuracy_over_dataset_size(
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

    N_iterations_list = list(df_data_dict.keys())
    N_patients_list = list(df_data_dict[N_iterations_list[0]].keys())
    
    for iteration in N_iterations_list:
            
        for N_patients in N_patients_list:
            df_UQ_temp = df_data_dict[iteration][N_patients]


            if ENDPOINT_TYPES[endpoint] == "Binary":
                df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Labels'] != -1]
                true_labels = df_UQ_no_missing_labels['True Labels'].values
            else:
                df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Label Event'] != -1]
                true_labels = df_UQ_no_missing_labels[['True Label Event', 'True Months Event']].values

            mean_preds = df_UQ_no_missing_labels['Mean Prediction'].values
            #print(df_UQ_temp)

            # threshold = 0.5
            fpr, tpr, thresholds = roc_curve(true_labels, mean_preds)
            idx = np.argmax(tpr - fpr) # use the threshold at the biggest difference between true positive rate and false positive rate
            threshold = thresholds[idx]
            
            accuracy = accuracy_score(true_labels, mean_preds > threshold)
            auc = roc_auc_score(true_labels, mean_preds)
            # accuracy = roc_auc_score(true_labels, mean_preds)
            rows.append({
                "endpoint": endpoint,
                "N_patients": int(N_patients),
                "accuracy": accuracy,
                'auc' : auc
            })

    
    df = pd.DataFrame(rows)

    grouped = df.groupby(['N_patients'])['accuracy'].agg(['mean', 'std']).reset_index()
    ax.errorbar(
        grouped['N_patients'],
        grouped['mean'],
        yerr=grouped['std'],
        label='Accuracy',
        marker='o',
        capsize=3
    )
    grouped_auc = df.groupby(['N_patients'])['auc'].agg(['mean', 'std']).reset_index()
    ax.errorbar(
        grouped_auc['N_patients'],
        grouped_auc['mean'],
        yerr=grouped_auc['std'],
        label='AUC',
        marker='o',
        capsize=3
    )
    

    ax.set_xlabel("N training patients")