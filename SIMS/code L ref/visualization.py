# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:53:31 2024

@author: HoekL02
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from torchmetrics.classification import BinaryCalibrationError
from sklearn.calibration import calibration_curve


def confusion_matrix_bin(true,pred,outpath):
    
    plt.figure()
    confusion_matrix = metrics.confusion_matrix(true, pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.savefig(outpath)
    plt.clf()
    plt.close('all')
    
    return

def loss_plot(loss_t,loss_v, n_epochs, outpath):
    
    plt.figure()
    plt.plot(list(range(n_epochs)),loss_t)
    plt.plot(list(range(n_epochs)),loss_v)
    plt.savefig(outpath)
    plt.clf()
    plt.close('all')
    
    return

def calibration_plot(true,pred,n_bins,outpath):
    
    prob_true, prob_pred = calibration_curve(true,pred, n_bins=n_bins)
    
    plt.figure()
    plt.scatter(prob_pred,prob_true)
    plt.plot(prob_pred,prob_true,'-')
    plt.plot([0,1],[0,1],'--', c = 'k')
    plt.xlabel("Predicted probability (mean confidence)")
    plt.ylabel("True occurance rate")
    plt.savefig(outpath)
    plt.clf()
    plt.close('all')
    
    return

