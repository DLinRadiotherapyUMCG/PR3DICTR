import numpy as np
import torch


def calc_bins(y_true, y_pred, num_bins = 10, bin_type="fixed"):
    """
    Divides the predictions into bins and calculates the accuracy, confidence and size of each bin.
    Bins can be either `fixed` or `adaptive`, where `fixed` bins are equally spaced and `adaptive` bins are equally populated.
    """
    # Assign each prediction to a bin
    if bin_type == "fixed":              # uses fixed bin widths (e.g. 0.1-0.2, 0.2-0.3, ..., 0.9-1.0, for ECE)
        bins = np.linspace(1/num_bins, 1, num_bins)
        binned = np.digitize(y_pred, bins, right=True)
    elif bin_type == "adaptive":         # adaptive bin widths, ensures equal number of samples in each bin (e.g. ACE metric)
        bins = np.linspace(1/num_bins, 1, num_bins)
        bins = np.quantile(y_pred, bins, axis=0)
        bins = np.unique(bins)
        bins[-1] = 1
        binned = np.digitize(y_pred, bins, right=True)
    else:
        raise ValueError("Invalid bin_type. Must be 'fixed' or 'adaptive'.")

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



def compute_ECE(y_true, y_pred, n_bins = 10):
    """
    Compute the ECE for a set of predictions and labels.
    """
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_true, y_pred, num_bins=n_bins, bin_type="fixed")

    ECE = 0

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif

    return ECE



def compute_MCE(y_true, y_pred, n_bins = 10):
    """
    Compute the MCE for a set of predictions and labels.
    """
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_true, y_pred, num_bins=n_bins, bin_type="fixed")

    MCE = 0
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        MCE = max(MCE, abs_conf_dif)

    return MCE




def compute_ACE(y_true, y_pred, n_bins = 10):
    """
    Compute the ECE for a set of predictions and labels.
    """
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_pred, y_true, num_bins=n_bins, bin_type="adaptive")

    ACE = 0

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ACE += abs_conf_dif
    ACE = ACE / n_bins

    return ACE




def calulate_calibration_metric_for_multiple_endpoints(config, y_true_list_dict, y_pred_list_dict, metric= 'ECE', n_bins=10):
    """
    Calculate the calibration metric for multiple endpoints.
    """
    metric_dict = {}
    for key in y_true_list_dict.keys():
        y_true = y_true_list_dict[key]
        y_pred = y_pred_list_dict[key]

        if isinstance(y_true, torch.Tensor) and y_true.is_cuda:
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

        # mask out missing labels
        mask = [True if x in config.valid_endpoint_values else False for x in y_true]
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # compute the metric of choice
        if metric == 'ECE':
            metric_dict[key] = compute_ECE(y_true, y_pred, n_bins)
        elif metric == 'MCE':
            metric_dict[key] = compute_MCE(y_true, y_pred, n_bins)
        elif metric == 'ACE':
            metric_dict[key] = compute_ACE(y_true, y_pred, n_bins)
        else:
            raise ValueError("Invalid metric. Must be 'ECE', 'MCE' or 'ACE'.")

    return metric_dict