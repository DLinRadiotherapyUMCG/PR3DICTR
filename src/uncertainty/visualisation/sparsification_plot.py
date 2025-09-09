import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score


def plot_sparsification_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, colours_dict):
    """
    Plots a sparsification subplot on the given axis. Sparsification involves iteratively removing samples with the highest uncertainty
    and evaluating the performance metric (AUC for binary classification, C-index for survival analysis) at each step.
    Args:
        ax: Matplotlib axis to plot on.
        df_UQ_temp: DataFrame containing the UQ results for a specific endpoint and method
        endpoint: The endpoint being evaluated (e.g., "OS", "LRC").
        ENDPOINT_TYPES: Dictionary mapping endpoints to their types ("Binary" or "Event").
        UQ_metrics_list: List of uncertainty metrics to evaluate (e.g., ["Predictive Entropy", "Mutual Information"]).
        colours_dict: Dictionary mapping UQ metric names to colors for plotting.
    Returns:
        None (the plot is drawn on the provided axis).
    """

    if ENDPOINT_TYPES[endpoint] == "Binary":
        df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Labels'] != -1]
        labels_iter = df_UQ_no_missing_labels['True Labels'].copy()
    else:
        df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Label Event'] != -1]
        labels_iter = df_UQ_no_missing_labels['True Label Event'].copy()
        months_iter = df_UQ_no_missing_labels['True Months Event'].copy()
        labels_iter = np.stack([labels_iter.values, months_iter.values], axis=1)

    mean_preds_iter = df_UQ_no_missing_labels['Mean Prediction'].copy()

    for UQ_metric_name in UQ_metrics_list:
        UQ_metric_iter = df_UQ_no_missing_labels[UQ_metric_name].copy()
        indices = np.arange(len(mean_preds_iter))
        all_metric_values = []

        labels_arr = labels_iter.copy()
        preds_arr = mean_preds_iter.copy()
        uq_arr = UQ_metric_iter.copy()

        # while np.unique(labels_arr if ENDPOINT_TYPES[endpoint] == "Binary" else labels_arr[:, 0]).size >= 2:
        for _ in range(len(mean_preds_iter)):
            max_idx = np.argmax(uq_arr)
            preds_arr = np.delete(preds_arr, max_idx, axis=0)
            labels_arr = np.delete(labels_arr, max_idx, axis=0)
            uq_arr = np.delete(uq_arr, max_idx, axis=0)
            indices = np.delete(indices, max_idx, axis=0)

            if ENDPOINT_TYPES[endpoint] == "Binary":
                if np.unique(labels_arr).size == 2:
                    
                    # metric_value = roc_auc_score(labels_arr, preds_arr)
                    thresh_value = 0.5
                    metric_value = accuracy_score(labels_arr, preds_arr>thresh_value)
                else:
                    metric_value = 1
            else:
                if np.unique(labels_arr[:, 0]).size == 2:
                    metric_value = concordance_index(labels_arr[:, 1], preds_arr, labels_arr[:, 0])
                else:
                    metric_value = 1
            all_metric_values.append(metric_value)

        ax.plot(all_metric_values, label=UQ_metric_name, color=colours_dict[UQ_metric_name])
        ax.set_xlim(0, len(all_metric_values) - 1)
        ax.set_ylim(0, 1.0)
    ax.set_xlabel("Number of Samples Dropped")