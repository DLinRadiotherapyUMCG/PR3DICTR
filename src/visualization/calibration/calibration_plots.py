import numpy as np
from src.evaluation.metrics.utils import calc_bins
from src.evaluation.metrics.utils import remove_missing, threshold




def get_bins_std(preds_or_labels, binned):
    bin_stds = []
    for i in range(binned.max() + 1):
        #print(binned == i)
        bin_preds = preds_or_labels[binned == i]
        bin_std = bin_preds.std()
        bin_stds.append(bin_std)
    return np.array(bin_stds)



def make_row_calibration_plots(config, row_ax, preds, labels, column_names, n_bins, colours):
    #print(preds)
    for i, col_name in enumerate(column_names):
        calibration_subplot(config, row_ax[i], preds[col_name], labels[col_name], colours[i], n_bins)




def calibration_subplot(config, ax, y_pred_all, y_true_all, colour, n_bins=10):

    y_true, y_pred = remove_missing(config, y_true_all, y_pred_all)

    bins, binned, bin_accs, bin_confs, bin_sizes = calc_bins(config, y_true, y_pred, bin_type="adaptive")
    
    coef = np.polyfit(bin_confs, bin_accs, 1)
    linear_fit = np.poly1d(coef)

    # plot the points (scatter)
    shape = 'o'
    #label = model_name

    ax.plot([0,1], [0,1], "black", label="Ideal") # plot the ideal line

    print(n_bins, len(bin_accs), len(bin_confs))
    
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





