import matplotlib.pyplot as plt

from src.uncertainty.visualisation.calibration_plot import plot_calibration_subplot, plot_error_calibration_subplot, plot_calibration_error_over_dataset_size, plot_UQ_values_over_dataset_size, plot_accuracy_over_dataset_size
from src.uncertainty.visualisation.sparsification_plot import plot_sparsification_subplot
from src.uncertainty.visualisation.correlation_plots import plot_uncertainty_and_error_correlation_subplot, plot_uncertainty_and_error_AUROC_subplot
from matplotlib.axis import YAxis


column_names_dict = {
    "Dysphagia_M06": "Dysphagia",
    "Xerostomia_M06": "Xerostomia",
    "OS_2year_censored" : "2 Year Survival",
    "LRC_2year_censored" : "2 Year Locoregional Control",
}


def plot_nested_UQ(
    UQ_RESULTS_DICT,
    ENDPOINT_TYPES,
    UQ_methods_list,
    UQ_metrics_list,
    plot_type="calibration", # "calibration" or "sparsification" or 'calibration_error_dataset_size
    row_key="method",
    col_key="endpoint",
    normalisation_method="minmax",
    N_bins=5,
    colours_dict=None,
    ):
    
    """
    Plots the nested uncertainty quantification (UQ) results. The function creates a grid of subplots based on the specified row and column keys (which can be "endpoint", "method", or "metric").
    Each subplot visualizes the relationship between uncertainty and performance metrics (AUC for binary classification, C-index for survival analysis) using different UQ metrics.
    Args:
        UQ_RESULTS_DICT: Dictionary containing UQ results for different endpoints and methods.
        ENDPOINT_TYPES: Dictionary mapping endpoints to their types ("Binary" or "Event").
        UQ_methods_list: List of UQ methods to include in the plots (e.g., ["MC Dropout", "Deep Ensemble"]).
        UQ_metrics_list: List of uncertainty metrics to evaluate (e.g., ["Predictive Entropy", "Mutual Information"]).
        plot_type: Type of plot to generate ("calibration", "sparsification", "calibration_error_dataset_size", "error calibration").
        row_key: Key to use for rows in the subplot grid ("endpoint", "method", or "metric").
        col_key: Key to use for columns in the subplot grid ("endpoint", "method", or "metric").
        normalisation_method: Method for normalizing uncertainty metrics ("minmax", "percentile", or None).
        N_bins: Number of bins to use for calibration plots.
        colours_dict: Optional dictionary mapping UQ metric names to colors for plotting.
    Returns:
        fig: Matplotlib figure containing the nested UQ plots.
    """
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
            
            if plot_type == "UQ_calibration" or plot_type == "prediction_calibration":
                plot_calibration_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, N_bins, colours_dict, normalisation_method=normalisation_method, plot_type=plot_type)
            elif plot_type == "error calibration":
                plot_error_calibration_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, N_bins, colours_dict, normalisation_method=normalisation_method)
            elif plot_type == "sparsification":
                plot_sparsification_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, colours_dict)
            elif plot_type == "calibration_error_dataset_size":
                plot_calibration_error_over_dataset_size(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, N_bins, colours_dict, normalisation_method=normalisation_method)
            elif plot_type == "plot_UQ_values_over_dataset_size":
                is_last_col = (j == n_cols - 1)
                plot_UQ_values_over_dataset_size(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, N_bins, colours_dict, normalisation_method=normalisation_method, is_last_col=is_last_col)
            elif plot_type == "plot_accuracy_over_dataset_size":
                plot_accuracy_over_dataset_size(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, N_bins, colours_dict, normalisation_method=normalisation_method)
            elif plot_type == "uncertainty_error_correlation":
                plot_uncertainty_and_error_correlation_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, colours_dict, normalisation_method=normalisation_method)
            elif plot_type == "uncertainty_and_error_AUROC":
                plot_uncertainty_and_error_AUROC_subplot(ax, df_UQ_temp, endpoint, ENDPOINT_TYPES, UQ_metrics_list, colours_dict, normalisation_method=normalisation_method)
            else:
                raise ValueError(f"Unknown plot_type: {plot_type}")
            

            # Titles and labels
            if i == 0 and n_rows >= 1:
                try:
                    col_val = column_names_dict[col_val]
                except KeyError:
                    pass
                ax.set_title(col_val)
            if j == 0:
                #ax.set_ylabel("AUC" if ENDPOINT_TYPES[endpoint] == "Binary" else "C-Index")
                if plot_type == "UQ_calibration" or plot_type == "sparsification":
                    ax.set_ylabel("Accuracy" if ENDPOINT_TYPES[endpoint] == "Binary" else "C-Index")
                elif plot_type == "prediction_calibration":
                    ax.set_ylabel("Observed Rate")
                elif plot_type == "calibration_error_dataset_size":
                    ax.set_ylabel("UQ Calibration Error")
                elif plot_type == "error calibration":
                    ax.set_ylabel("Error (BCE loss)")
                elif plot_type == "plot_UQ_values_over_dataset_size":
                    ax.set_ylabel("Mean Uncertainty")
                ax.annotate(row_val, xy=(-0.3, 0.5), xycoords='axes fraction',
                            ha='right', va='center', fontsize=12, rotation=90)
            else:
                ax.yaxis.set_visible(False)
            
            # Hide xlabel for all axes except those in the last row
            if i != n_rows - 1:
                ax.set_xlabel("")  # hide the xlabel for all but the last row
                # pass  # xlabel is set in subfunctions
            
            # if i == 0 and j == 0:
            #     fig.legend(title="UQ Metric", loc='upper right', bbox_to_anchor=(1.1, 0.92))



    if plot_type == "UQ_calibration":
        title = None # "Certainty vs Accuracy"

        # title = "Calibration Plot for Pharynx Patients"

    elif plot_type == "prediction_calibration":
        title = "Prediction vs Observed Rate"
    elif plot_type == "sparsification":
        title = "Number of 'Most Uncertain' Samples Removed vs. Accuracy"
    elif plot_type == "calibration_error_dataset_size":
        title = "Training Dataset Size vs. UQ Calibration Error"
    elif plot_type == "error calibration":
        title = "Uncertainty vs Error"
    elif plot_type == "plot_UQ_values_over_dataset_size":
        title = "Training Dataset Size vs. Mean Uncertainty"
    elif plot_type == "plot_accuracy_over_dataset_size":
        title = "Training Dataset Size vs. Classification Performance"
    elif plot_type == "uncertainty_error_correlation":
        title = "Uncertainty vs Error Scatter Plot"
    elif plot_type == "uncertainty_and_error_AUROC":
        title = "AUROC of uncertainty vs. correctness"


    plt.suptitle(
        title,
        fontsize=16
    )

    #plt.tight_layout()
    
    return fig

# Example usage:
# fig = plot_nested_UQ(UQ_RESULTS_DICT, ENDPOINT_TYPES, UQ_methods_list, UQ_metrics_list, plot_type="calibration", row_key="method", col_key="endpoint")
# fig = plot_nested_UQ(UQ_RESULTS_DICT, ENDPOINT_TYPES, UQ_methods_list, UQ_metrics_list, plot_type="sparsification", row_key="endpoint", col_key="method")
