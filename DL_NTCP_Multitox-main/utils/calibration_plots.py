import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.calibration import calibration_curve
from misc import make_colorlist


def adaptive_make_calibration_plots(config, row_dicts, column_names, title=None, mode='calibration', filedir=None, return_fig=False):
    """
    Make calibration plots for multiple models. This code is flexible in how many rows and columns the diagram can have

    """

    ECE_dicts_list, MCE_dicts_list = [], []
    colours = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:pink', 'tab:purple', 'tab:brown', 'tab:cyan']


    # set the dimensions of the plot
    row_count = len(row_dicts)
    column_count = len(column_names)

    plot_width, plot_height = 5, 5

    fig_width = plot_width * column_count 
    fig_height = plot_height * row_count

    if mode == "calibration":
        #title = "Calibration Plot: {}".format(set_name)
        x_axis_label = "Predicted Rate"
        y_axis_label = "Observed Rate"
        
    elif mode == "reliability":
        #title = "Reliability Plot: {}".format(set_name)
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
            make_row_calibration_plots(config, axs[i], row_preds, row_labels, column_names, 10, colours, model_name)
        else:
            ECE_dict, MCE_dict = make_row_reliability_plots(config, axs[i], row_preds, row_labels, column_names, 10, colours, model_name)
            ECE_dicts_list.append(ECE_dict)
            MCE_dicts_list.append(MCE_dict)


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
        fig.savefig(filedir, bbox_inches='tight')

    if return_fig:
        return fig
    plt.close()

    if mode == 'reliability':
        return ECE_dicts_list, MCE_dicts_list
    



def calc_bins(y_pred, y_true, num_bins = 10):
    # Assign each prediction to a bin
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(y_pred, bins, right=True)
    #unique, counts = np.unique(binned, return_counts=True)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    labels_oneh = y_true
    for bin in range(num_bins):
        bin_sizes[bin] = len(y_pred[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (y_pred[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def get_ECE_metrics(bins, _, bin_accs, bin_confs, bin_sizes):
    """
    Calculates the ECE and MCE metrics for a set of bins.
    """
    ECE = 0
    MCE = 0

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE







def make_row_calibration_plots(config, row_ax, preds, labels, column_names, n_bins, colours, model_name):
    #print(preds)
    for i, col_name in enumerate(column_names):
    #    print(i)
    #for ax, i in enumerate(row_ax):
        calibration_subplot(config, row_ax[i], preds[col_name].cpu().numpy(), labels[col_name].cpu().numpy(), colours[i], model_name, n_bins)


def make_row_reliability_plots(config, row_ax, preds, labels, column_names, n_bins, colours, model_name):
    ECE_dict = {}
    MCE_dict = {}
    for i, col_name in enumerate(column_names):
        ECE, MCE = reliability_subplot(config, row_ax[i], preds[col_name].cpu().numpy(), labels[col_name].cpu().numpy(), n_bins, colours[i], model_name)
        ECE_dict[col_name] = ECE
        MCE_dict[col_name] = MCE

    return ECE_dict, MCE_dict
    

def reliability_subplot(config, ax, y_pred_all, y_true_all, n_bins, colour, model_name):
    mask = [True if x in config.valid_endpoint_values else False for x in y_true_all]

    y_true = y_true_all[mask]
    y_pred = y_pred_all[mask]

    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_pred, y_true, num_bins=n_bins)

    ECE, MCE = get_ECE_metrics(bins, _, bin_accs, bin_confs, bin_sizes)

    bin_centers = bins - 0.05

    # Error bars (light red bars)
    ax.bar(bin_centers, bin_centers,  width=0.1, alpha=0.2, edgecolor='black', color='r', hatch='/', zorder=1)

    # Draw (blue) bars and identity line
    ax.bar(bin_centers, bin_accs, width=0.1, alpha=1, edgecolor='black', color=colour, zorder=2)
    ax.plot([0,1],[0,1], '--', color='gray', linewidth=2, zorder=3)

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
    ax.legend(handles=[ECE_patch, MCE_patch])

    # Create grid
    ax.grid(True, color='gray', linestyle='dashed', zorder=0)

    # plot gaps
    gaps = (bin_accs - bin_centers) 
    gaps = np.clip(gaps, 0, 1)

    #gaps = gaps if gaps > 0 else 0
    ax.bar(bin_centers, gaps, bottom = np.minimum(bin_accs, (bins-0.05)),
                width=0.1, alpha=0.4, edgecolor='black', color='r', hatch='\\/', zorder=3)
    
    ECE = round(ECE, config.nr_of_decimals)
    MCE = round(MCE, config.nr_of_decimals)

    return ECE, MCE


def calibration_subplot(config, ax, y_pred_all, y_true_all, colour, model_name, n_bins=10):
    
    x = [0,1]
    y = [0,1]

    mask = [True if x in config.valid_endpoint_values else False for x in y_true_all]

    y_true = y_true_all[mask]
    y_pred = y_pred_all[mask]

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, pos_label=1, n_bins=n_bins, strategy="quantile")

    coef = np.polyfit(mean_predicted_value, fraction_of_positives, 1)
    linear_fit = np.poly1d(coef)

    # plot the points (scatter)
    shape = 'o'
    label = model_name

    ax.plot([0, 1], [0, 1], "black", label="Ideal") # plot the ideal line

    ax.plot(mean_predicted_value, fraction_of_positives, shape, label=label, color=colour)
    # plot the best fit line
    ax.plot(x, linear_fit(y), '--',   color=colour)
    
    ax.legend()

