# -*- coding: utf-8 -*-
"""
Helper functions for classification metrics
"""
import numpy as np
from sklearn.metrics import roc_curve



def remove_missing(config, true, pred):
    """
    Removes the values that are indicated as missing with the missing indicator -1
    """
    missing_val = config['data']['missing_data_value']

    pred = pred[np.where(true != missing_val)]
    true = true[np.where(true != missing_val)]

    return true, pred



def threshold(config, true, pred):
    """
    Sets the threshold to split the data, currently only compatable with binairy endpoints
    """
    threshold = config['evaluation']['metrics']['threshold']
    
    if threshold == 'YoudenJ':
        fpr, tpr, thresholds = roc_curve(true, pred)
        idx = np.argmax(tpr - fpr) # use the threshold at the biggest difference between true positive rate and false positive rate
        threshold = thresholds[idx]
        
    elif type(threshold) == float:
        threshold = threshold
        
    else:
        print('non_proper threshold selection, defaulted to 0.5')
        threshold = 0.5

    # the actual thresholding    
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0

    return true, pred



def calc_bins(config, y_true, y_pred, bin_type="fixed"):
    """
    Divides the predictions into bins and calculates the accuracy, confidence and size of each bin.
    Bins can be either `fixed` or `adaptive`, where `fixed` bins are equally spaced and `adaptive` bins are equally populated.
    """
    #print(config['evaluation']['metrics'])
    num_bins = config['evaluation']['metrics']['calibration_bins']
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
