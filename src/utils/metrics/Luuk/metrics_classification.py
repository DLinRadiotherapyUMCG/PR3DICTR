# -*- coding: utf-8 -*-
"""
Classification metrics
"""
import numpy as np
from metric_utils import remove_missing, threshold
from sklearn.metrics import  accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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