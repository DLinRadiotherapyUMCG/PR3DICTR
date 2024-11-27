# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import metrics

def confusion_matrix_bin(true,pred,outpath):
    
    plt.figure()
    confusion_matrix = metrics.confusion_matrix(true, pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.savefig(outpath)
    plt.clf()
    plt.close('all')
    
    return