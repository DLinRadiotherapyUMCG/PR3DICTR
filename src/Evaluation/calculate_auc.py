import logging
import math

import numpy as np
from sklearn.metrics import roc_auc_score

import torch

def sigmoid(x):
    return 1/(1+np.exp(-x))

def calculate_auc(outputs, targets):
    if np.unique(targets.cpu().detach().numpy()).size > 1:
        auc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
    else:
        auc = None
        logging.warning(f'Cannot compute ROC AUC score because targets contain only one class')
    return auc


def calculate_auc_multi(output,targets,config):
    
    labels = config['columns']['label']  
    predictions = output 
    out_dict = {}
    Sigmoid = np.vectorize(sigmoid)


    for i in labels:
        temp_out = []
        temp_target = []

        for ii in range(len(predictions[i])):
            if(targets[i][ii] != -1 or math.isnan(predictions[i][ii])):
                temp_out_value = float(Sigmoid(predictions[i][ii]))
                if(math.isnan(temp_out_value)):
                    print(f"sigmoid generate nan values for {predictions[i][ii]}")
                else:
                    temp_out.append(temp_out_value)
                    temp_target.append(targets[i][ii])
        try:
            auc = roc_auc_score(temp_target, temp_out)
        except Exception as e:
            print(f"An error occured during AUC: type = {type(e)}, args = {e.args}, message = {e}")
            print("Additional info for AUC calculation:")
            print(temp_target)
            print(temp_out)
            auc = -1

        out_dict[i] = auc
    
    return out_dict