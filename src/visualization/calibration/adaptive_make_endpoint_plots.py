import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# import all the necessary plotting functions
from src.visualization.calibration.kaplan_meier import make_row_kaplan_meier_plots
from src.visualization.calibration.calibration_plots import make_row_calibration_plots
from src.visualization.calibration.reliability_plots import make_row_reliability_plots
from src.visualization.calibration.confusion_matrix_plots import make_row_confusion_matrix_plots
from src.visualization.calibration.roc_curve_plots import make_row_ROC_plots



def adaptive_make_endpoint_plots(config, row_dicts, column_names, title=None, mode='calibration', filedir=None, return_fig=False):
    """
    Make calibration plots for multiple models. This code is flexible in how many rows and columns the diagram can have

    """
    n_bins = config['evaluation']['visualisations']['n_bins']
    colours = ['tab:pink',  'tab:orange', 'tab:green', 'tab:grey', 'tab:blue', 'tab:orange', 'tab:brown', 'tab:cyan']


    # set the dimensions of the plot
    row_count = len(row_dicts)
    column_count = len(column_names)

    plot_width, plot_height = 4, 4

    fig_width = plot_width * column_count 
    fig_height = plot_height * row_count

    if mode == "calibration":
        x_axis_label = "Predicted Rate"
        y_axis_label = "Observed Rate"
        
    elif mode == "reliability":
        x_axis_label = "Confidence"
        y_axis_label = "Accuracy"
    elif mode == "confusion_matrix":
        x_axis_label = "Predicted"
        y_axis_label = "True"
    elif mode == "roc_curve":
        x_axis_label = "False Positive Rate"
        y_axis_label = "True Positive Rate"
    elif mode == "kaplan_meier":
        date_unit = config['data']['event_endpoint_time_unit'].capitalize()
        x_axis_label = f"Time ({date_unit})"
        y_axis_label = "Rate"
    else:
        raise ValueError("Calibration plotting mode must be either `calibration` or `reliability` or 'confusion_matrix' or 'roc_curve'")


    # Create a figure and axis
    # Create a gridspec to add a colorbar axis
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        row_count,
        column_count,
        width_ratios=[1] * column_count,
        height_ratios=[1] * row_count,
        hspace=0.1,
        wspace=0.1,
        figure=fig,
    )

    # Create title
    if title is not None:
        title_height = 0.92 if len(row_dicts) > 1 else 1
        fig.suptitle(title, fontsize=20, y=title_height) # place the title at the top of the figure

    # Create subplots
    axs = [
        [plt.subplot(gs[i, j]) for j in range(column_count)] for i in range(row_count)
    ]

    for i, row_pred_dict in enumerate(row_dicts):
        row_preds = row_pred_dict['preds']
        row_labels = row_pred_dict['labels']
        
        if mode == "calibration":
            make_row_calibration_plots(config, axs[i], row_preds, row_labels, column_names, n_bins, colours)
        elif mode == 'reliability':
            make_row_reliability_plots(config, axs[i], row_preds, row_labels, column_names, n_bins, colours)
        elif mode == 'confusion_matrix':
            make_row_confusion_matrix_plots(config, axs[i], row_preds, row_labels, column_names)
        elif mode == 'roc_curve':
            make_row_ROC_plots(config, axs[i], row_preds, row_labels, column_names)
        elif mode == 'kaplan_meier':
            make_row_kaplan_meier_plots(config, axs[i], row_preds, row_labels, column_names)

        

    # set the axis labels and ticks for all subplots
    for i in range(row_count):
        for j in range(column_count):
            #axs[i][j].set_aspect("equal")
            if i != row_count - 1:
                axs[i][j].set_xticklabels([])
                axs[i][j].set_frame_on(True)
                axs[i][j].tick_params(tick1On=False)
            else:
                axs[i][j].set_xlabel(x_axis_label)

            if i == 0:
                axs[i][j].set_title(column_names[j]) # annotate column subtitle
            
            if j != 0:
                axs[i][j].set_yticklabels([])
                axs[i][j].set_frame_on(True)
                axs[i][j].tick_params(tick1On=False)
            else:   
                axs[i][j].set_ylabel(y_axis_label)
                axs[i][j].annotate(         # annotate row subtitle
                    row_dicts[i]["name"],
                    xy=(0, 0.5),
                    xytext=(-axs[i][j].yaxis.labelpad - 50, 0),
                    xycoords=axs[i][j].yaxis.label,
                    textcoords="offset points",
                    size=15,
                    ha="center",
                    va="center",
                )

            if mode not in ['confusion_matrix', 'kaplan_meier']:   # set the axis limits for all subplots. do not do this for the confusion matrices, it breaks them
                axs[i][j].set_xlim(0, 1)
                axs[i][j].set_ylim(0, 1)

    if filedir is not None:
        os.makedirs(Path(filedir).parent, exist_ok=True)
        fig.savefig(filedir, bbox_inches='tight')

    if return_fig:
        return fig
    plt.close()

    