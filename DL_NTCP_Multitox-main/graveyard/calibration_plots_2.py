import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.calibration import calibration_curve
from misc import make_colorlist

def make_calibration_plots(config, predictions_dict, true_labels_dict, lr_predictions_dict, lr_true_labels_dict, set_name, mode = "calibration", filename=None):
    """
    Makes a calivration plot using a set predicitons and true labels. The predictions are split into 10 bins for this.
    The plot is then saved to a file if a `filename` is provided.
    Args:
        config: a ConfigObject.
        predictions_dict: a dictionary of lists of predictions for each endpoint.
        true_labels_dict: a dictionary of lists of true labels for each endpoint.
        set_name: a string, the name of the set (e.g. 'train', 'val', 'test') to be used in the title of the plot.
        filename: a string, the name of the file to save the plot to.
    Returns:
        None
    """
    if lr_predictions_dict == None:
        #return None, None, None, None
        DL_ECE = {endpoint : list() for endpoint in config.endpoint_list}
        DL_MCE = {endpoint : list() for endpoint in config.endpoint_list}
        LR_ECE = {endpoint : list() for endpoint in config.endpoint_list}
        LR_MCE = {endpoint : list() for endpoint in config.endpoint_list}
    
        #return DL_ECE, DL_MCE, LR_ECE, LR_MCE
    
    #colours = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:pink', 'tab:purple', 'tab:brown', 'tab:cyan']
    colours = make_colorlist(config)
    #colours = ['blue', 'orange', 'red', 'green', 'magenta', 'darkviolet', 'saddlebrown', 'cyan']

    DL_ECE = {}
    DL_MCE = {}
    LR_ECE = {}
    LR_MCE = {}

    num_endpoints = len(config.endpoint_list)
    row_count = 2 if lr_predictions_dict is not None else 1
    fig_width = 5 * num_endpoints 
    fig_height = 5 * row_count

    # Create a gridspec to add a colorbar axis
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        row_count,
        num_endpoints,
        width_ratios=[1] * num_endpoints,
        hspace=0.1,
        wspace=0.1,
        figure=fig,
    )

    # Create title
    if mode == "calibration":
        title = "Calibration Plot: {}".format(set_name)
        x_axis_label = "Predicted Rate"
        y_axis_label = "Observed Rate"
        
    elif mode == "reliability":
        title = "Reliability Plot: {}".format(set_name)
        x_axis_label = "Confidence"
        y_axis_label = "Accuracy"
    else:
        raise ValueError("Calibration plotting mode must be either `calibration` or `reliability`")
    
    fig.suptitle(title, fontsize=15, y=0.95)

    # Create subplots
    axs = [
        [plt.subplot(gs[i, j]) for j in range(num_endpoints)] for i in range(row_count)
    ]

    labels = ["DL\nModel", "CITOR\nModels"]

    for i in range(row_count):
        for j in range(num_endpoints):
            axs[i][j].set_aspect("equal")
            if i != row_count - 1:
                axs[i][j].set_xticklabels([])
                axs[i][j].set_frame_on(True)
                axs[i][j].tick_params(tick1On=False)
            else:
                axs[i][j].set_xlabel(x_axis_label)
            
            if j != 0:
                axs[i][j].set_yticklabels([])
                axs[i][j].set_frame_on(True)
                axs[i][j].tick_params(tick1On=False)
            else:
                axs[i][j].set_ylabel(y_axis_label)
                axs[i][j].annotate(
                    labels[i],
                    xy=(0, 0.5),
                    xytext=(-axs[i][j].yaxis.labelpad - 40, 0),
                    xycoords=axs[i][j].yaxis.label,
                    textcoords="offset points",
                    size=15,
                    ha="center",
                    va="center",
                )

            axs[i][j].set_xlim(0, 1)
            axs[i][j].set_ylim(0, 1)


    for idx, (ax, endpoint) in enumerate(zip(axs[0], config.endpoint_list)):
        ax.set_title(endpoint, fontdict={'fontsize': 13})

        DL_preds = predictions_dict[endpoint].clone().cpu().numpy()
        DL_true = true_labels_dict[endpoint].clone().cpu().numpy()

        if mode == "calibration":
            calibration_subplot(config, ax, DL_preds, DL_true, color=colours[idx], DL_model=True)
        elif mode == "reliability":
            ece, mce = draw_reliability_subplot(config, ax, DL_preds, DL_true, color=colours[idx], DL_model=True)
            DL_ECE[endpoint] = ece
            DL_MCE[endpoint] = mce
    
    if lr_predictions_dict is not None:
        for idx, (ax, endpoint) in enumerate(zip(axs[1], config.endpoint_list)):
            LR_preds = lr_predictions_dict[endpoint].clone().cpu().numpy()
            LR_true = lr_true_labels_dict[endpoint].clone().cpu().numpy()

            if mode == "calibration":
                calibration_subplot(config, ax, LR_preds, LR_true, color=colours[idx], DL_model=False)
            elif mode == "reliability":
                ece, mce = draw_reliability_subplot(config, ax, LR_preds, LR_true, color=colours[idx], DL_model=False)
                LR_ECE[endpoint] = ece
                LR_MCE[endpoint] = mce
    else:
        for endpoint in config.endpoint_list:
            LR_ECE[endpoint] = None
            LR_MCE[endpoint] = None

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        #plt.savefig(filename)
    #plt.show()
    plt.close(fig)

    return DL_ECE, DL_MCE, LR_ECE, LR_MCE





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




def calibration_subplot(config, ax, y_pred_all, y_true_all, color, DL_model = True, n_bins=10):
    """
    Draws the UMCG-style calibration plot
    """
    x = [0,1]
    y = [0,1]

    mask = [True if x in config.valid_endpoint_values else False for x in y_true_all]

    y_true = y_true_all[mask]
    y_pred = y_pred_all[mask]

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, pos_label=1, n_bins=n_bins, strategy="quantile")

    coef = np.polyfit(mean_predicted_value, fraction_of_positives, 1)
    linear_fit = np.poly1d(coef)

    # plot the points (scatter)
    shape = 'o' if DL_model else 's'
    label = "DL Model" if DL_model else "CITOR"

    ax.plot([0, 1], [0, 1], "black", label="Ideal") # plot the ideal line

    ax.plot(mean_predicted_value, fraction_of_positives, shape, label=label, color=color)
    # plot the best fit line
    ax.plot(x, linear_fit(y), '--',   color=color)
    
    ax.legend()



def draw_reliability_subplot(config, ax, y_pred_all, y_true_all, color, DL_model = True, n_bins=10):
    """
    Draws the ECE reliability subplot for a set of predictions and true labels.
    """
    mask = [True if x in config.valid_endpoint_values else False for x in y_true_all]

    y_true = y_true_all[mask]
    y_pred = y_pred_all[mask]

    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_pred, y_true, num_bins=n_bins)

    ECE, MCE = get_ECE_metrics(bins, _, bin_accs, bin_confs, bin_sizes)

    bin_centers = bins - 0.05

    # Error bars (light red bars)
    ax.bar(bin_centers, bin_centers,  width=0.1, alpha=0.2, edgecolor='black', color='r', hatch='/', zorder=1)

    # Draw (blue) bars and identity line
    ax.bar(bin_centers, bin_accs, width=0.1, alpha=1, edgecolor='black', color=color, zorder=2)
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

