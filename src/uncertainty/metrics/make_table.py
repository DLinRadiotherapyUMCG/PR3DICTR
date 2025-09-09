import pandas as pd
import numpy as np

from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from lifelines.utils import concordance_index

def make_nested_UQ_table(
        config,
    UQ_RESULTS_DICT,
    ENDPOINT_TYPES,
    UQ_methods_list,
    UQ_metrics_list,
    metric_func,  # function to compute the metric, e.g., compute_ece
    N_bins,
):
    # Prepare rows for DataFrame
    rows = []
    for endpoint in UQ_RESULTS_DICT.keys():
            
        for method in UQ_methods_list:
            df_UQ_temp = UQ_RESULTS_DICT.get(endpoint, {}).get(method, None)

            if ENDPOINT_TYPES[endpoint] == "Binary":
                df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Labels'] != -1]
                true_labels = df_UQ_no_missing_labels['True Labels'].values
            else:
                df_UQ_no_missing_labels = df_UQ_temp[df_UQ_temp['True Label Event'] != -1]
                true_labels = df_UQ_no_missing_labels[['True Label Event', 'True Months Event']].values

            mean_preds = df_UQ_no_missing_labels['Mean Prediction'].values
            
            if df_UQ_temp is None:
                continue
            for uq_metric_name in UQ_metrics_list:
                        
                UQ_metric = df_UQ_no_missing_labels[uq_metric_name].values
                if uq_metric_name == "Binary Entropy":
                    UQ_metric_norm = UQ_metric
                else:
                    UQ_metric_norm = (UQ_metric - np.min(UQ_metric)) / (np.max(UQ_metric) - np.min(UQ_metric) + 1e-8)

                uq_bins = pd.qcut(UQ_metric_norm, N_bins, labels=False, duplicates='drop')
                aucs, bin_centers = [], []
                for b in np.unique(uq_bins):
                    idx = uq_bins == b
                    if ENDPOINT_TYPES[endpoint] == "Binary":
                        if np.unique(true_labels[idx]).size == 2:
                            # auc = roc_auc_score(true_labels[idx], mean_preds[idx])
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
                    bin_centers.append(1 - UQ_metric_norm[idx].mean())
                
                value = np.sum((np.array(aucs) - np.array(bin_centers))**2) / len(aucs)
                # COMPUTE METRIC HERE
                # value = metric_func(config, aucs, bin_centers)
                rows.append({
                    "endpoint": endpoint,
                    "method": method,
                    "uq_metric": uq_metric_name,
                    "metric_value": value
                })
    df = pd.DataFrame(rows)
    return df
