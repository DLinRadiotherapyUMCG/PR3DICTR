# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:38:51 2024

@author: HoekL02
"""

import matplotlib.pyplot as plt
from sklearn import metrics

def ROC_curve(true,pred,config, label):
    fpr, tpr, _ = metrics.roc_curve(true,  pred)
    auc = metrics.roc_auc_score(true, pred)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.title('ROC curve ' + label)
    plt.legend(loc=4)
    plt.savefig(config['paths']['results'] + 'ROC_curve.png')
    plt.close()
    
    return