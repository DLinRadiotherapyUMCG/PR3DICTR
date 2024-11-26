# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def confusion_matrix_multi(nameFile, output, targets, config):
    
    labels = config['columns']['label']  
    predictions = output 
    Sigmoid = np.vectorize(sigmoid)


    for i in labels:
        temp_out = []
        temp_target = []

        for ii in range(len(predictions[i])):
            if targets[i][ii] != -1:
                temp_out.append(float(Sigmoid(predictions[i][ii])))
                temp_target.append(targets[i][ii])
        
        #try:
        confusion_matrix_bin(nameFile, np.array(temp_target), np.array(temp_out), config)
        #except Exception as e:
        #    print(f"Error: The confusion matrix could not be created due to: {e}")
        #    pass
    return

def confusion_matrix_bin(true, pred,config, label, show = False):
    
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
    plt.title('Confusion matrix ' + label)
    if(show == False):
        plt.savefig(config['paths']['results'] + label + '_confusion_matrix.png')
        plt.clf()
        plt.close('all')
    
    return