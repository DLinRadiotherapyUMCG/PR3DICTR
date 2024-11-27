import os
from pathlib import Path
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.calibration import calibration_curve
from src.utils.metrics.misc import make_colorlist
from src.utils.metrics.ECE import calc_bins, compute_ECE, compute_MCE

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
    else:
        raise ValueError("Calibration plotting mode must be either `calibration` or `reliability`")


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
        title_height = 0.92 if len(row_dicts) > 1 else 0.96
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
        else:
            make_row_reliability_plots(config, axs[i], row_preds, row_labels, column_names, n_bins, colours, model_name)       

    # set the axis labels and ticks for all subplots
    for i in range(row_count):
        for j in range(column_count):
            axs[i][j].set_aspect("equal")
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

            axs[i][j].set_xlim(0, 1)
            axs[i][j].set_ylim(0, 1)

    if filedir is not None:
        os.makedirs(Path(filedir).parent,exist_ok=True)
        fig.savefig(filedir, bbox_inches='tight')

    if return_fig:
        return fig
    plt.close()
    



def make_row_calibration_plots(config, row_ax, preds, labels, column_names, n_bins, colours, model_name):
    #print(preds)
    for i, col_name in enumerate(column_names):
        try:
    #    print(i)
    #for ax, i in enumerate(row_ax):
            calibration_subplot(config, row_ax[i], preds[col_name].cpu().numpy(), labels[col_name].cpu().numpy(), colours[i], model_name, n_bins)
        except Exception as e:
            print(e)
            pass


def make_row_reliability_plots(config, row_ax, preds, labels, column_names, n_bins, colours, model_name):
    for i, col_name in enumerate(column_names):
        try:
            reliability_subplot(config, row_ax[i], preds[col_name].cpu().numpy(), labels[col_name].cpu().numpy(), n_bins, colours[i], model_name)
        except:
            pass
    

def reliability_subplot(config, ax, y_pred_all, y_true_all, n_bins, colour, model_name):
    valid_endpoint_values = [0,1]
    mask = [True if x in valid_endpoint_values else False for x in y_true_all]

    y_true = y_true_all[mask]
    y_pred = y_pred_all[mask]

    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_true, y_pred, num_bins=n_bins)

    ECE = compute_ECE(y_true, y_pred, n_bins)
    MCE = compute_MCE(y_true, y_pred, n_bins)

    # set the width of the bins, and the centers of the bins (both are dependent on the number of bins)
    bin_width = 1 / n_bins
    bin_centers = bins - (bin_width / 2)

    # Error bars (light red bars)
    ax.bar(bin_centers, bin_centers, width=bin_width, alpha=0.2, edgecolor='black', color='r', hatch='/', zorder=1)

    # Draw (blue) bars and identity line
    ax.bar(bin_centers, bin_accs, width=bin_width, alpha=1, edgecolor='black', color=colour, zorder=2)
    ax.plot([0,1],[0,1], '--', color='gray', linewidth=2, zorder=3)

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}'.format(ECE))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}'.format(MCE))
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

    bins, binned, bin_accs, bin_confs, bin_sizes = calc_bins(y_true, y_pred, num_bins = n_bins, bin_type="adaptive")
    
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

