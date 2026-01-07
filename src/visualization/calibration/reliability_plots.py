import matplotlib.patches as mpatches
import numpy as np
from src.evaluation.metrics.utils import calc_bins
from src.evaluation.metrics.calibration import ECE, MCE
from src.evaluation.metrics.utils import remove_missing, threshold




def make_row_reliability_plots(config, row_ax, preds, labels, column_names, n_bins, colours):
    for i, col_name in enumerate(column_names):
        reliability_subplot(config, row_ax[i], preds[col_name], labels[col_name], n_bins, colours[i])
       


def reliability_subplot(config, ax, y_pred_all, y_true_all, n_bins, colour):

    y_true, y_pred = remove_missing(config, y_true_all, y_pred_all)

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