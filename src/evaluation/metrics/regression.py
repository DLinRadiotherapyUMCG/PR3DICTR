"""
Regression metrics
"""
from src.evaluation.metrics.utils import remove_missing
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

def r2(config, true, pred): # R squared
    true, pred = remove_missing(config,true,pred)
    return r2_score(true, pred)

def MSE(config, true, pred): # Mean Squared error
    true, pred = remove_missing(config,true,pred)
    return mean_squared_error(true, pred)

def MAE(config, true, pred): # Mean absolute error
    true, pred = remove_missing(config,true,pred)
    return mean_absolute_error(true, pred)

def RMSE(config, true, pred): # root mean sqaured error
    true, pred = remove_missing(config,true,pred)
    return root_mean_squared_error(true, pred)