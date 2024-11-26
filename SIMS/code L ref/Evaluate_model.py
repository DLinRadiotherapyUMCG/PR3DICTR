# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:59:17 2024

@author: Luuk.vanderHoek
"""

"""
Goals:
    1. Load information from pickle file and load model parameters to evaluate 
        on (new) data
    
    
    2. Load information from pickle file and be able to train a model using 
        those hyperparameters



"""

import joblib
from misc import save_predictions


def load_optuna(study, weights, model = "DCNN"):
    """
    Input:
        optuna study .pkl, model weight location, model function type
        
    function: 
        1. Generate model architecture based on optuna study parameters
        2. Load model weights into the model
        
    returns:
        model with weights
    
    """
    
    model = ''
    return model






if __name__ == '__main__':
    
    a = joblib.load()