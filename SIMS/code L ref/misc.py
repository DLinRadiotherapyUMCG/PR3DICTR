# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:53:31 2024

@author: HoekL02
"""

from data_preparation import get_umcg_n,Data_split,get_patiens,get_endpoint_data,PatRated_survey_data

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

#%% model evaluation 

def save_predictions(model,timepoint,out_path = '', device = 'cuda:0'):
    
    # device = 'cuda:' + torch.cuda.current_device()
    
    sm = nn.Softmax(dim = 1)
    
    umcg_n = get_umcg_n()
    # umcg_n = umcg_n[:100]
    train,validation,test = Data_split(umcg_n)
    bindf, orddf,um = PatRated_survey_data()
    
    preds = []
    label = []
    imputed = []
    sets = []
    IDs = []
    
    tot = 0
    
    model.eval() 
    with torch.no_grad():
        for n in umcg_n:
            if n in train or n in test or n in validation:
                inp = get_patiens([n])
                preds.append(sm(model(torch.tensor(inp).cuda(device),None))[0][1].cpu().detach().numpy())
                label.append(get_endpoint_data(np.array([n]), time_point_idx = timepoint, imputed = True, binary = True, tox = 'xerostomia')[1])
                imputed.append(np.isnan(bindf[:,timepoint][np.where(um == n)[0]]))
                IDs.append(n)
            if n in train:
                sets.append("train")                 
            if n in validation:
                sets.append("validation")                   
            if n in test:
                sets.append("test")           
            tot+=1
            print(str(tot) + '//' + str(len(umcg_n)))
    
    df = pd.DataFrame({'IDs':umcg_n,'Prediction':preds,'label':label,'imputed': imputed}) # Potential additions: tumor location
    
    df.to_excel(out_path)
    
    return

def evaluate_model(model,data,loss= nn.CrossEntropyLoss(label_smoothing=0)):
    sm = nn.Softmax()
    
    model.eval() 
    with torch.no_grad():
        running_loss = 0 
        c = 0
        pred = []
        true = []
        pred_l = []
        for a,b in enumerate(data):
            inp = b[0].cuda()
            label = b[1].type(torch.LongTensor).cuda()
            output = model(inp,None)
            output = sm(output)
            ls = loss(output,label)
            running_loss += ls.cpu().detach().numpy()
            c += np.sum((torch.argmax(output,dim =1) == label).cpu().detach().numpy())    
            pred.extend(torch.argmax(output,dim =1).cpu().detach().numpy())
            pred_l.extend(output[:,1].cpu().detach().numpy())
            true.extend(label.cpu().detach().numpy())
                         
    return pred, pred_l, true, running_loss, c


#%% Early stopping ideas
def compount_metric(loss, AUC = None, F1 = None, ECE = None, acc = None, W = [.4,.15,.15,.15,.15]):
    return loss*W[0]+AUC*W[1]+F1*W[2]+ECE*W[3]+acc*W[4]

def compound_evaluator(loss, AUC = None, F1 = None, ECE = None, acc = None, patience = 5,
                       W = [.4,.15,.15,.15,.15]):
    """
    
    Calculates early stopping criteria based on a compound of evaluation metrics
    
    
    Returns: Stop 
    
    """
    
    stop = False
    
    compound = loss*W[0]+AUC*W[1]+F1*W[2]+ECE*W[3]+acc*W[4]
    
    if compound[-patience] == np.min(compound[-patience:]):
        stop = True
    
    return stop

def early_stopping(loss, AUC = None, F1 = None, ECE = None, acc = None, compound = None, patience = 10, method = 'loss'):
     """
     input:
         type of early stopping criteria: loss, compound or AUC

    returns: 
        Boolean, if you should stop the training procedure
    
     """
    
     stop = False
     
     if method == 'AUC':
         if AUC[-patience] == np.max(AUC[-patience:]):
             stop = True
        
     if method == 'compound':
        if compound == None:
           stop = compound_evaluator(loss,AUC,F1,ECE,acc, patience = patience)
        elif compound[-patience] == np.min(compound[-patience:]):
            stop = True
            
     if method == 'loss':
         try:
             if loss[-patience] == np.min(loss[-patience:]):
                 stop = True
         except:
             stop = False
    
     return stop

