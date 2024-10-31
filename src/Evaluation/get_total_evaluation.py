# -*- coding: utf-8 -*-
"""
Get the total evaluation of a model,

Things it can contain:
    
    General:
        - Per patient prediction 
        - Per patient uncertainty (Entropy, aleatoric, epistemic) (optional WIP, later)
        - Per patient logistic regression prediction (WIP)
    
    Visualization:
        - Confusion matrix 
        - Calibration plot
        - attention/ atribution maps (optional WIP, later)
        - patient visualzation (optional WIP, later)
        
    Metrics:
        - AUC (Area Under the Curve)
        - ECE (Expected Calibration Error)
        - Accuracy, precision, sensitivity... (Based on Youden J thresholds)
    

I/O:
    I: 
        - model location
    O: 
        - csv: predictions, uncertainty, metrics
        - png: confusion matrix, calibration plot
        - npy: attention/ atribution maps, patient visualization
"""

import numpy as np
import torch
import logging
import os
import pandas as pd

from src.constants import DEVICE

from src.config_presets.tools.load_config import load_config
from src.dataset.load_dataset import load_dataset_total
from src.models.tools.get_multi_model import get_classification_model
from src.visualization.get_visualization import get_visualization
from src.evaluation.calculate_auc import calculate_auc_multi


from torch.utils.data import DataLoader

def sigmoid(x):
    return 1/(1+np.exp(-x))

def move_batch_to_device(batch, device):
    inputs, clinical_features, targets = batch
    inputs = inputs.to(device=device)
    clinical_features = clinical_features.to(device=device)
    targets = targets.to(device=device)
    return inputs, clinical_features, targets

def get_output_results(model, val_loader, config): # this function is redundant, could just be an option for the validate function
    model.eval()
    labels = config['columns']['label']
      
    out_tot = dict.fromkeys(labels)
    targets_tot = dict.fromkeys(labels)

    for label in labels:
        out_tot[label] = []
        targets_tot[label] = []
    
    Sigmoid = np.vectorize(sigmoid)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            logging.debug(f'Validation batch {i}')
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

            outputs = model(x=inputs, features=clinical_features)

            lab_indx = 0
            for label in labels:
                out_tot[label] = out_tot[label] + list(Sigmoid(outputs[label].cpu().detach().numpy().reshape((1,targets[:,:,lab_indx].shape[0]))[0]))
                targets_tot[label] = targets_tot[label] + list(targets[:,:,lab_indx].cpu().detach().numpy().reshape((1,targets[:,:,lab_indx].shape[0]))[0])
                lab_indx+=1
    return(out_tot, targets_tot)

def get_total_evaluation(model_path, savePath = ""):
    
    # Check if there are fold models in the model location
    folds = os.listdir(model_path)
    # folds = folds.sort()
    print(folds)

    # Load a config to get model architecture information
    path = os.path.join(os.path.join(model_path,folds[0]),"DlModel_Config.yaml")
    print(path)
    config = load_config(path)#model_path + r'/' + folds[0] + '/DlModel_Config.yaml')
    
    # Load the testing dataset
    data, metadata = load_dataset_total(config)
    test_data = DataLoader(data[2][0],batch_size=config['training']['batch_size'], shuffle=False, 
                                      num_workers = config['data']['dataloader']['num_workers'], persistent_workers = config['data']['dataloader']['persistent_workers'])
    
    # Get the ensemble predictions
    out_lists = []
    tar_lists  = []
    
    i = 0
    for path_fold in folds:   
        model = get_classification_model(config, metadata)
        model.cuda()
        model.load_state_dict(torch.load(model_path + r'/' + path_fold + '/DlModel_Weights.pth'))
        out, targets = get_output_results(model, test_data, config)

        out_lists.append(out)
        tar_lists.append(targets)
        if i == 0:
            out_tot = dict.fromkeys(out.keys())
            for key in out_tot.keys():
                out_tot[key] = np.array(out[key])*(1/len(folds))
            i+=1
        else:
            for key in out_tot.keys():
                out_tot[key] = out_tot[key] + np.array(out[key])*(1/len(folds))       
        
    # General
    if(savePath == ""):
        pd.DataFrame(out_tot).to_csv(config['paths']['results'] + 'predictions.csv')
        pd.DataFrame(targets).to_csv(config['paths']['results'] + 'labels.csv')
    else:
        os.makedirs(savePath)
        pd.DataFrame(out_tot).to_csv(os.path.join(savePath, 'predictions.csv'))
        pd.DataFrame(targets).to_csv(os.path.join(savePath, 'labels.csv'))
    
    # Visualization 
    # for key in out_tot.keys():
    #     true = targets[key]
    #     pred = out_tot[key]
    #     get_visualization(config,true,pred, key)
    
    # # Metrics
    # auc = calculate_auc_multi()
    # ECE = calulate_ECE_multi() # TO DO
    # acc, sens, spec, F1 = calculate_acc_metrics()
    
    
    
    return 