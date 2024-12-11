import os
from pathlib import Path
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib import gridspec
from sklearn.calibration import calibration_curve
#from src.utils.metrics.misc import make_colorlist
#from src.eval.metrics.ECE import calc_bins, compute_ECE, compute_MCE
from src.evaluation.metrics.utils import calc_bins
from src.evaluation.metrics.calibration import ECE, MCE
from src.evaluation.metrics.utils import remove_missing, threshold


def adaptive_make_calibration_plots(config, row_dicts, column_names, title=None, mode='calibration', n_bins=10, filedir=None, return_fig=False):
    """
    Make calibration plots for multiple models. This code is flexible in how many rows and columns the diagram can have

    """

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
    else:
        raise ValueError("Calibration plotting mode must be either `calibration` or `reliability` or 'confusion_matrix' or 'roc_curve'")


    # Create a figure and axis
    # Create a gridspec to add a colorbar axis
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        row_count,
        column_count,
        width_ratios=[1] * column_count,
        hspace=0,
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
        model_name = row_pred_dict['name']
        row_preds = row_pred_dict['preds']
        row_labels = row_pred_dict['labels']

        
        if mode == "calibration":
            make_row_calibration_plots(config, axs[i], row_preds, row_labels, column_names, n_bins, colours, model_name)
        elif mode == 'reliability':
            make_row_reliability_plots(config, axs[i], row_preds, row_labels, column_names, n_bins, colours, model_name)
        elif mode == 'confusion_matrix':
            make_row_confusion_matrix_plots(config, axs[i], row_preds, row_labels, column_names, n_bins, colours, model_name)
        elif mode == 'roc_curve':
            make_row_ROC_plots(config, axs[i], row_preds, row_labels, column_names, n_bins, colours, model_name)
        

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
                axs[i][j].set_title(column_names[j])   # annotate column subtitle
            
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

            if mode not in ['confusion_matrix']:   # set the axis limits for all subplots. do not do this for the confusion matrices, it breaks them
                axs[i][j].set_xlim(0, 1)
                axs[i][j].set_ylim(0, 1)

    if filedir is not None:
        os.makedirs(Path(filedir).parent, exist_ok=True)
        fig.savefig(filedir, bbox_inches='tight')

    if return_fig:
        return fig
    plt.close()
    



def make_row_calibration_plots(config, row_ax, preds, labels, column_names, n_bins, colours, model_name):
    #print(preds)
    for i, col_name in enumerate(column_names):
        calibration_subplot(config, row_ax[i], preds[col_name], labels[col_name], colours[i], model_name, n_bins)



def make_row_reliability_plots(config, row_ax, preds, labels, column_names, n_bins, colours, model_name):
    for i, col_name in enumerate(column_names):
        reliability_subplot(config, row_ax[i], preds[col_name], labels[col_name], n_bins, colours[i], model_name)
       


def reliability_subplot(config, ax, y_pred_all, y_true_all, n_bins, colour, model_name):
    valid_endpoint_values = [0,1]
    mask = [True if x in valid_endpoint_values else False for x in y_true_all]

    y_true = y_true_all[mask]
    y_pred = y_pred_all[mask]

    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(config, y_true, y_pred, bin_type="fixed")

    ECE_score = ECE(config, y_true, y_pred)
    MCE_score = MCE(config, y_true, y_pred)

    # set the width of the bins, and the centers of the bins (both are dependent on the number of bins)
    bin_width = 1 / n_bins
    bin_centers = bins - (bin_width / 2)

    # Error bars (light red bars)
    ax.bar(bin_centers, bin_centers, width=bin_width, alpha=0.2, edgecolor='black', color='r', hatch='/', zorder=1)

    # Draw (blue) bars and identity line
    ax.bar(bin_centers, bin_accs, width=bin_width, alpha=1, edgecolor='black', color=colour, zorder=2)
    ax.plot([0,1],[0,1], '--', color='gray', linewidth=2, zorder=3)

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}'.format(ECE_score))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}'.format(MCE_score))
    ax.legend(handles=[ECE_patch, MCE_patch])

    # Create grid
    ax.grid(True, color='gray', linestyle='dashed', zorder=0)

    # plot gaps
    gaps = (bin_accs - bin_centers) 
    gaps = np.clip(gaps, 0, 1)

    #gaps = gaps if gaps > 0 else 0
    ax.bar(bin_centers, gaps, bottom = np.minimum(bin_accs, bin_centers),
                width=bin_width, alpha=0.4, edgecolor='black', color='r', hatch='\\/', zorder=3)
    



def get_bins_std(preds_or_labels, binned):
    bin_stds = []
    for i in range(binned.max() + 1):
        #print(binned == i)
        bin_preds = preds_or_labels[binned == i]
        bin_std = bin_preds.std()
        bin_stds.append(bin_std)
    return np.array(bin_stds)


def calibration_subplot(config, ax, y_pred_all, y_true_all, colour, model_name, n_bins=10):
    valid_endpoint_values = [0,1]
    mask = [True if x in valid_endpoint_values else False for x in y_true_all]

    y_true = y_true_all[mask]
    y_pred = y_pred_all[mask]

    #fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, pos_label=1, n_bins=n_bins, strategy="quantile")

    bins, binned, bin_accs, bin_confs, bin_sizes = calc_bins(config, y_true, y_pred, bin_type="adaptive")
    
    coef = np.polyfit(bin_confs, bin_accs, 1)
    linear_fit = np.poly1d(coef)

    # plot the points (scatter)
    shape = 'o'
    #label = model_name

    ax.plot([0,1], [0,1], "black", label="Ideal") # plot the ideal line
    
    #ax.plot(mean_predicted_value, fraction_of_positives, shape, label=label, color=colour)
    ax.plot(bin_confs, bin_accs, shape, color=colour)
    #ax.hist(bin_confs, bins=n_bins)
    ax.hist(y_pred, weights=np.ones_like(y_pred)/len(y_pred), alpha=0.4, bins=np.maximum(10,n_bins))
    """
    NOTE: these three lines are to compute and plot the error bars. The error bars are not currently being plotted
    as the error bar along the y-axis are really long (because they are for the observed rate, not the predicted rate)
    """
    # bin_acc_stds = get_bins_std(y_true, binned=binned)
    # bin_conf_stds = get_bins_std(y_pred, binned=binned)
    #ax.errorbar(bin_confs, bin_accs, yerr=bin_acc_stds, xerr=bin_conf_stds, fmt=shape, color=colour)

    # plot the best fit line (from the lowest prediction to the highest prediction)
    x = [bin_confs.min(), bin_confs.max()]
    y = [bin_confs.min(), bin_confs.max()]
    ax.plot(x, linear_fit(y), '--',   color=colour)
    
    ax.legend()







### CONFUSION MATRIX PLOTTING

def make_row_confusion_matrix_plots(config, row_ax, preds, labels, column_names, n_bins, colours, model_name):
    for i, col_name in enumerate(column_names):
        #confusion_matrix_subplot(config, row_ax[i], preds[col_name], labels[col_name])


        labels2, preds2 = remove_missing(labels[col_name], preds[col_name])
        labels_thresholded, preds_thresholded = threshold(config, labels2, preds2)

        confusion_matrix = metrics.confusion_matrix(labels_thresholded, preds_thresholded)

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        cm_display.plot(ax=row_ax[i], colorbar=False, include_values=True)

        if i!=0:
            cm_display.ax_.set_ylabel('')
        



### ROC CURVE PLOTTING

def make_row_ROC_plots(config, row_ax, preds, labels, column_names, n_bins, colours, model_name):
    for i, col_name in enumerate(column_names):
        fpr, tpr, _ = metrics.roc_curve(labels[col_name], preds[col_name])
        auc = metrics.roc_auc_score(labels[col_name], preds[col_name])

        row_ax[i].plot([0,1],[0,1], '--', color='gray', linewidth=2, zorder=3)  # ideal line
        row_ax[i].plot(fpr,tpr,label="AUC="+str(auc))









