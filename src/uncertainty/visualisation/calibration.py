from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd

from src.evaluation.metrics.classification import accuracy


def plot_UQ_calibration(UQ_RESULTS_DICT, ENDPOINT_TYPES, UQ_methods_list, UQ_metrics_list):
    endpoint_list = list(UQ_RESULTS_DICT.keys())

    n_endpoints = len(endpoint_list)
    n_methods = len(UQ_methods_list)
    n_metrics = len(UQ_metrics_list)
    
    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colours_dict = {UQ_metric_name: colour for UQ_metric_name, colour in zip(UQ_metrics_list, colours)}

    fig, axes = plt.subplots(n_methods, n_endpoints, figsize=(3 * n_endpoints, 3 * n_methods), sharey=True, sharex=True)

    for col_idx, endpoint in enumerate(endpoint_list):
        for row_idx, UQ_method_name in enumerate(UQ_methods_list):
        # Robustly get the correct axis for any shape of axes
            if n_endpoints == 1 and n_methods == 1:
                ax = axes
            elif n_endpoints == 1:
                ax = axes[col_idx]
            elif n_methods == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]

        # plot the ideal line here
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Calibration')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            try:
                df_UQ_temp = UQ_RESULTS_DICT[endpoint][UQ_method_name]
            except KeyError:
                print(f"Warning: No data for endpoint '{endpoint}' and UQ method '{UQ_method_name}'. Skipping this plot.")
                continue
        
            if ENDPOINT_TYPES[endpoint] == "Binary":
                df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Labels'] != -1]
                true_labels = df_UQ_no_missing_labels['True Labels'].values
            else:
                df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Label Event'] != -1]
            # For event-based endpoints, we use the True Label Event and True Months Event
                true_labels = df_UQ_no_missing_labels[['True Label Event', 'True Months Event']].values
            
            mean_preds = df_UQ_no_missing_labels['Mean Prediction'].values
        
            N_bins = 5  # Number of bins for uncertainty metric

            for UQ_metric_name in UQ_metrics_list:
                UQ_metric = df_UQ_no_missing_labels[UQ_metric_name].values
            # Normalize the UQ metric to [0, 1] using min-max normalization
            
                if UQ_metric_name == "Binary Entropy":
                    UQ_metric_norm = UQ_metric  # binary entropy is already in [0, 1]
                else:  
                    UQ_metric_norm = (UQ_metric - np.min(UQ_metric)) / (np.max(UQ_metric) - np.min(UQ_metric)) # min-max normalization

            # Bin the UQ metric into 10 bins
                binning_method = "quantile"

                if binning_method == "quantile":
                    uq_bins = pd.qcut(UQ_metric_norm, N_bins, labels=False, duplicates='drop')
                elif binning_method == "uniform":
                    uq_bins = pd.cut(UQ_metric_norm, N_bins, labels=False, include_lowest=True, duplicates='drop')
                else:
                    raise ValueError(f"Unknown binning_method: {binning_method}")
                aucs = []
                bin_centers = []

                for b in np.unique(uq_bins):
                #print(b)
                    idx = uq_bins == b
                # Only calculate AUC if both classes are present in the bin
                    if ENDPOINT_TYPES[endpoint] == "Binary": 
                        if np.unique(true_labels[idx]).size == 2:
                            auc = roc_auc_score(true_labels[idx], mean_preds[idx])
                        #auc = accuracy(config, true_labels[idx], mean_preds[idx])
                            aucs.append(auc)
                        else:
                            aucs.append(1)   # all labels in the bin are the same, so AUC is 1
                    else:
                        if np.unique(true_labels[idx][:, 0]).size == 2:
                            auc = concordance_index(true_labels[idx][:, 1], mean_preds[idx], true_labels[idx][:, 0])
                        #auc = roc_auc_score(true_labels[idx], mean_preds[idx])
                            aucs.append(auc)
                        else:
                            aucs.append(1)
                
                    bin_centers.append(1 - UQ_metric_norm[idx].mean())

                shape = 'o'
                ax.plot(bin_centers, aucs, shape, label=UQ_metric_name, color=colours_dict[UQ_metric_name])

            # plot the best-fit line
                coef = np.polyfit(bin_centers, aucs, 1)
                linear_fit = np.poly1d(coef)
                ax.plot(bin_centers, linear_fit(bin_centers), linestyle='-', color=colours_dict[UQ_metric_name], alpha=0.8, linewidth=2)


            if row_idx == 0:
                ax.set_title(endpoint)
            if row_idx == n_methods - 1:
                ax.set_xlabel("Certainty (1 - uncertainty)")
            
            if row_idx == 0 and col_idx == 0:
                fig.legend(title="UQ Metric", loc='upper right', bbox_to_anchor=(1.2, 0.98))

        # Annotate the endpoint name to the left of each row
            if col_idx == 0:
                ax.set_ylabel("AUC")
                ax.annotate(UQ_method_name, xy=(-0.3, 0.5), xycoords='axes fraction',
                        ha='right', va='center', fontsize=12, rotation=90)

    plt.suptitle("AUC vs. Uncertainty Bins for Different UQ Methods", fontsize=16)
    plt.tight_layout()
    #plt.show()


    return fig