# -*- coding: utf-8 -*-
"""
File containing all required metric functions

(temporarily?) in  a single file

I hoped to take the metrics mostly form the scikit-learn package

"""

#%% imports

import numpy as np
from sklearn.metrics import  r2_score, brier_score_loss, accuracy_score, balanced_accuracy_score,roc_curve, f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

#%% Helper functions
def remove_missing(true, pred):
    """
    Removes the values that are indicated as missing with the missing indicator -1
    """
    pred = pred[np.where(true != -1)]
    true = true[np.where(true != -1)]
    return true, pred

def threshold(true, pred, config):
    """
    Sets the threshold to split the data, currently only compatable with binairy endpoints
    """
    threshold = config['metrics']['threshold']
    
    if threshold == 'youdenJ':
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

def calc_bins(y_true, y_pred, config, bin_type="fixed"):
    """
    Divides the predictions into bins and calculates the accuracy, confidence and size of each bin.
    Bins can be either `fixed` or `adaptive`, where `fixed` bins are equally spaced and `adaptive` bins are equally populated.
    """
    num_bins = config['metrics']['calibration_bins']
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


#%% Classification
def balanced_acc(true, pred, config):
    true, pred = remove_missing(true,pred)
    true, pred = threshold(true, pred, config)
    return balanced_accuracy_score(true, pred)

def acc(true, pred, config):
    true, pred = remove_missing(true,pred)
    true, pred = threshold(true, pred, config)
    return accuracy_score(true, pred)

def F1(true, pred, config):
    true, pred = remove_missing(true,pred)
    true, pred = threshold(true, pred, config)
    return f1_score(true,pred)

def precision(true, pred, config):
    true, pred = remove_missing(true,pred)
    true, pred = threshold(true, pred, config)
    return precision_score(true,pred)

def recall(true, pred, config):
    true, pred = remove_missing(true,pred)
    true, pred = threshold(true, pred, config)
    return recall_score(true,pred)

def auc(true, pred):
    true, pred = remove_missing(true,pred)
    return roc_auc_score(true,pred)

def auc_se(true, pred):
    "from: https://real-statistics.com/descriptive-statistics/roc-curve-classification-table/auc-confidence-interval/"
    AUC = auc(true,pred)
    q0 = AUC*(1-AUC)
    q1 = AUC/(2-AUC) - AUC**2 
    q2 = (2*AUC**2)/(1+AUC) - AUC**2
    n1 = np.sum(true == 1)
    n2 = np.sum(true == 0)
    return np.sqrt((q0+(n1-1)*q1 +(n2-1)*q2)/(n1*n2))

#%% Regression
def r2(true, pred):
    true, pred = remove_missing(true,pred)
    return r2_score(true, pred)

def mse(true, pred):
    true, pred = remove_missing(true,pred)
    return mean_squared_error(true, pred)

def mae(true, pred):
    true, pred = remove_missing(true,pred)
    return mean_absolute_error(true, pred)

def rmse(true, pred):
    true, pred = remove_missing(true,pred)
    return root_mean_squared_error(true, pred)

#%% Calibration
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
