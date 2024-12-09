# -*- coding: utf-8 -*-
"""
Regression metrics
"""
from metric_utils import remove_missing
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

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