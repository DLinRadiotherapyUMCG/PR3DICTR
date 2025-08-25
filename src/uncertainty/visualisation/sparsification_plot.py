import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index


def sparsification_plot(UQ_RESULTS_DICT, ENDPOINT_TYPES, UQ_methods_list, UQ_metrics_list):
    endpoint_list = list(UQ_RESULTS_DICT.keys())

    n_endpoints = len(endpoint_list)
    n_methods = len(UQ_methods_list)
    n_metrics = len(UQ_metrics_list)

    fig, axes = plt.subplots(n_methods, n_endpoints, figsize=(3*len(endpoint_list), 3 * len(UQ_methods_list)), sharey=True, sharex=False)

    for col_idx, endpoint in enumerate(endpoint_list):
        #for col_idx, (ax, UQ_metric_name) in enumerate(zip(row, UQ_metrics_list)):
        for row_idx, UQ_method_name in enumerate( UQ_methods_list):

            # Robustly get the correct axis for any shape of axes
            if n_endpoints == 1 and n_methods == 1:
                ax = axes
            elif n_endpoints == 1:
                ax = axes[col_idx]
            elif n_methods == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]

            if row_idx == 0:
                ax.set_title(endpoint)
            
            if row_idx == n_methods - 1:
                ax.set_xlabel("Number of Samples Dropped")
            if col_idx == 0:
                ax.set_ylabel("AUC" if ENDPOINT_TYPES[endpoint] == "Binary" else "C-Index")
                # Annotate the endpoint name to the left of each row
                ax.annotate(UQ_method_name, xy=(-0.3, 0.5), xycoords='axes fraction',
                    ha='right', va='center', fontsize=12, rotation=90)

            try:
                df_UQ_temp = UQ_RESULTS_DICT[endpoint][UQ_method_name].copy()
            except KeyError:
                print(f"Warning: No data for endpoint '{endpoint}' and UQ method '{UQ_method_name}'. Skipping this plot.")
                continue

            
            for UQ_metric_name in UQ_metrics_list:

                if ENDPOINT_TYPES[endpoint] == "Binary":
                    df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Labels'] != -1]

                    labels_iter = df_UQ_no_missing_labels['True Labels'].copy()
                else:
                    df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Label Event'] != -1]
                    # For event-based endpoints, we use the True Label Event and True Months Event
                    labels_iter = df_UQ_no_missing_labels['True Label Event'].copy()
                    months_iter = df_UQ_no_missing_labels['True Months Event'].copy()

                    labels_iter = np.stack([labels_iter.values, months_iter.values], axis=1)

                    #print(df_UQ_no_missing_labels)

                mean_preds_iter = df_UQ_no_missing_labels['Mean Prediction'].copy()

                UQ_metric_iter = df_UQ_no_missing_labels[UQ_metric_name].copy()
                indices = np.arange(len(mean_preds_iter))
                all_metric_values = []

                #print(endpoint, UQ_method_name, roc_auc_score(labels_iter, mean_preds_iter))

                while np.unique(labels_iter).size >= 2:
                    max_idx = np.argmax(UQ_metric_iter)
                    mean_preds_iter = np.delete(mean_preds_iter, max_idx, axis=0)
                    labels_iter = np.delete(labels_iter, max_idx, axis=0)
                    UQ_metric_iter = np.delete(UQ_metric_iter, max_idx, axis=0)
                    indices = np.delete(indices, max_idx, axis=0)
                    
                    if ENDPOINT_TYPES[endpoint] == "Binary":
                        if np.unique(labels_iter).size == 2:
                            metric_value = roc_auc_score(labels_iter, mean_preds_iter)
                        else:
                            metric_value = 1
                    else:
                        
                        if np.unique(labels_iter[:, 0]).size == 2:
                            metric_value = concordance_index(labels_iter[:, 1], mean_preds_iter, labels_iter[:, 0])
                        #auc = roc_auc_score(labels_iter, mean_preds_iter)
                        else:
                            metric_value = 1
                    all_metric_values.append(metric_value)


                ax.plot(all_metric_values, label=UQ_metric_name)
                ax.set_xlim(0, len(all_metric_values) - 1)
                ax.set_ylim(0, 1.0)

                # x = compute_s_auc(np.linspace(0, 1, len(all_metric_values)), all_metric_values)
                # print(x)
            if row_idx == 0 and col_idx == 0:
                fig.legend(title="UQ Metric", loc='upper right', bbox_to_anchor=(1.2, 0.98))

            # if row_idx == 0 and col_idx == 0:
            #         ax.legend(title="UQ Metric", loc='upper left')


    plt.suptitle("AUC vs. Samples Dropped for Different UQ Methods", fontsize=16)
    plt.tight_layout()
    
    return fig




def compute_s_auc(fractions_dropped, aucs, normalize=True):
    """
    Compute sparsification AUC (s-AUC).
    Parameters:
    - fractions_dropped: list or array of fractions of data removed (e.g., [0.0, 0.05, ..., 0.9])
    - aucs: list or array of AUCs computed after each fraction dropped
    - normalize: if True, normalize by the max possible area (so s-AUC is between 0 and 1)
    Returns:
    - s-AUC value (float)
    """
    assert len(fractions_dropped) == len(aucs), "Arrays must be same length"
    area = np.trapz(aucs, x=fractions_dropped)
    
    if normalize:
        max_area = np.trapz([1.0] * len(aucs), x=fractions_dropped)
        return area / max_area
    
    return area