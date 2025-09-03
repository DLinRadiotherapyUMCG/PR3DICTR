import matplotlib.pyplot as plt

from src.uncertainty.visualisation.calibration_plot import plot_calibration_subplot, plot_calibration_error_over_dataset_size
from src.uncertainty.visualisation.sparsification_plot import plot_sparsification_subplot



def plot_nested_UQ(
    UQ_RESULTS_DICT,
    ENDPOINT_TYPES,
    UQ_methods_list,
    UQ_metrics_list,
    plot_type="calibration", # "calibration" or "sparsification" or 'calibration_error_dataset_size
    row_key="method",
    col_key="endpoint",
    N_bins=5,
    colours_dict=None
):
    # Map keys to axes
    key_map = {
        "endpoint": list(UQ_RESULTS_DICT.keys()),
        "method": UQ_methods_list,
        "metric": UQ_metrics_list
    }
    row_keys = key_map[row_key]
    col_keys = key_map[col_key]

    n_rows, n_cols = len(row_keys), len(col_keys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharey=True, sharex=False)

    if colours_dict is None:
        colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        colours_dict = {metric: colour for metric, colour in zip(UQ_metrics_list, colours)}

    for i, row_val in enumerate(row_keys):
        for j, col_val in enumerate(col_keys):
            # Get axis robustly
            if n_rows == 1 and n_cols == 1:
                ax = axes
            elif n_rows == 1:
                ax = axes[j]
            elif n_cols == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]

            # Get endpoint/method depending on axes assignment
            endpoint = row_val if row_key == "endpoint" else col_val if col_key == "endpoint" else None
            method = row_val if row_key == "method" else col_val if col_key == "method" else None

            # Skip if endpoint/method not found
            if endpoint is None or method is None:
                ax.set_visible(False)
                continue
            
            try:
                df_UQ_temp = UQ_RESULTS_DICT[endpoint][method]
            except KeyError:
                ax.set_visible(False)
                continue
            
            if plot_type == "calibration":
                plot_calibration_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, N_bins, colours_dict)
            elif plot_type == "sparsification":
                plot_sparsification_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, colours_dict)
            elif plot_type == "calibration_error_dataset_size":
                plot_calibration_error_over_dataset_size(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, N_bins, colours_dict)
            

            # Titles and labels
            if i == 0:
                ax.set_title(col_val)
            if j == 0:
                #ax.set_ylabel("AUC" if ENDPOINT_TYPES[endpoint] == "Binary" else "C-Index")
                if plot_type == "calibration" or plot_type == "sparsification":
                    ax.set_ylabel("AUC" if ENDPOINT_TYPES[endpoint] == "Binary" else "C-Index")
                elif plot_type == "calibration_error_dataset_size":
                    ax.set_ylabel("UQ Calibration Error (MSE)")
                ax.annotate(row_val, xy=(-0.3, 0.5), xycoords='axes fraction',
                            ha='right', va='center', fontsize=12, rotation=90)
            
            if i != n_rows - 1:
                ax.set_xlabel("") # hide the xlabel for all but the last row
                pass  # xlabel is set in subfunctions
            if i == 0 and j == 0:
                fig.legend(title="UQ Metric", loc='upper right', bbox_to_anchor=(1.2, 0.98))

    plt.suptitle(
        "AUC vs. Uncertainty Bins for Different UQ Methods" if plot_type == "calibration"
        else "AUC vs. Samples Dropped for Different UQ Methods",
        fontsize=16
    )

    plt.tight_layout()
    
    return fig

# Example usage:
# fig = plot_nested_UQ(UQ_RESULTS_DICT, ENDPOINT_TYPES, UQ_methods_list, UQ_metrics_list, plot_type="calibration", row_key="method", col_key="endpoint")
# fig = plot_nested_UQ(UQ_RESULTS_DICT, ENDPOINT_TYPES, UQ_methods_list, UQ_metrics_list, plot_type="sparsification", row_key="endpoint", col_key="method")
