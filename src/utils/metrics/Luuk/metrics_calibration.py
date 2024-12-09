# -*- coding: utf-8 -*-
"""
calibration metrics
"""

from metric_utils import remove_missing, calc_bins
from sklearn.metrics import brier_score_loss

def brier(true, pred):
    true, pred = remove_missing(true,pred)
    return brier_score_loss(true, pred)

def ECE(true, pred, config):
    """
    Compute the ECE for a set of predictions and labels.
    """
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(true, pred, config, bin_type="fixed")

    ECE = 0
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    return ECE

def MCE(true, pred, config):
    """
    Compute the MCE for a set of predictions and labels.
    """
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(true, pred, config, bin_type="fixed")

    MCE = 0
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        MCE = max(MCE, abs_conf_dif)
    return MCE

def ACE(true, pred, config):
    """
    Compute the ECE for a set of predictions and labels.
    """
    num_bins = config['metrics']['calibration_bins']
    
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(pred, true, config, bin_type="adaptive")

    ACE = 0
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ACE += abs_conf_dif
    ACE = ACE / num_bins
    return ACE