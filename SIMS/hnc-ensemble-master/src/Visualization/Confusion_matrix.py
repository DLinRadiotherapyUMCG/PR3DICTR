# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
import numpy as np

def confusion_matrix_bin(true,pred,config):
    
    threshold = config['visualization']['confusion_matrix']['threshold']
    
    if threshold == 'YoudenJ':
        fpr, tpr, thresholds = roc_curve(true, pred)
        idx = np.argmax(tpr - fpr)
        threshold = thresholds[idx]
    
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0
    
    plt.figure()
    confusion_matrix = metrics.confusion_matrix(true, pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.savefig(config['paths']['results'] + 'confusion_matrix.png')
    plt.clf()
    plt.close('all')
    
    return