"""
Classification metrics
"""
import numpy as np
from src.evaluation.metrics.utils import remove_missing, threshold
from sklearn.metrics import  accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def balanced_acc(config, true, pred): # Balanced accuracy
    true, pred = threshold(config, true, pred)
    return balanced_accuracy_score(true, pred)

def accuracy(config, true, pred):
    true, pred = threshold(config, true, pred)
    return accuracy_score(true, pred)

def F1_score(config, true, pred): 
    true, pred = threshold(config, true, pred)
    return f1_score(true,pred)

def precision(config, true, pred):
    true, pred = threshold(config, true, pred)
    return precision_score(true,pred)

def recall(config, true, pred):
    true, pred = threshold(config, true, pred)
    return recall_score(true,pred)

def auc(config, true, pred, sample_weights=None): # Area under the curve
    auc = roc_auc_score(true,pred, sample_weight=sample_weights)
    return float(auc)
    #return roc_auc_score(true,pred, sample_weight=sample_weights)

def auc_se(config, true, pred): # standard error of area under the curve
    "from: https://real-statistics.com/descriptive-statistics/roc-curve-classification-table/auc-confidence-interval/"
    AUC = auc(true,pred)
    q0 = AUC*(1-AUC)
    q1 = AUC/(2-AUC) - AUC**2 
    q2 = (2*AUC**2)/(1+AUC) - AUC**2
    n1 = np.sum(true == 1)
    n2 = np.sum(true == 0)
    return np.sqrt((q0+(n1-1)*q1 +(n2-1)*q2)/(n1*n2))