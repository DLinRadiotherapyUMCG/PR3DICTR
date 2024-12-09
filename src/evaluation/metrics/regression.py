# -*- coding: utf-8 -*-
"""
Regression metrics
"""
from src.evaluation.metrics.utils import remove_missing
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

def r2(config, true, pred):
    true, pred = remove_missing(true,pred)
    return r2_score(true, pred)

def MSE(config, true, pred):
    true, pred = remove_missing(true,pred)
    return mean_squared_error(true, pred)

def MAE(config, true, pred):
    true, pred = remove_missing(true,pred)
    return mean_absolute_error(true, pred)

def RMSE(config, true, pred):
    true, pred = remove_missing(true,pred)
    return root_mean_squared_error(true, pred)