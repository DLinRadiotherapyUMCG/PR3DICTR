import logging

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
            if targets[i][ii] != -1:
                temp_out.append(float(Sigmoid(predictions[i][ii])))
                temp_target.append(targets[i][ii])
        
        try:
            auc = roc_auc_score(temp_target, temp_out)
        except:
            auc = -1
        out_dict[i] = auc
    
    return out_dict