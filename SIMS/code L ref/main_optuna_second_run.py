1# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:53:31 2024

@author: HoekL02
"""
from torch.utils.data import DataLoader
from resnet_relu import get_resnet_lrelu
from data_preparation import get_data_loader, get_umcg_n,Data_split,PatRated_survey_data,get_patiens,get_endpoint_data
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from DCNN_pooling import DCNN_Pooling

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import roc_curve as roc

from misc import early_stopping

import numpy as np

from methods import DCNN_LReLU, AdaBound

from visualization import loss_plot, confusion_matrix_bin, calibration_plot

import pandas as pd

import joblib

def save_predictions(model,timepoint,out_path = '', device = 'cuda:1'):
    
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
            inp = b[0].cuda('cuda:1')
            label = b[1].type(torch.LongTensor).cuda('cuda:1')
            output = model(inp,None)
            output = sm(output)
            ls = loss(output,label)
            running_loss += ls.cpu().detach().numpy()
            c += np.sum((torch.argmax(output,dim =1) == label).cpu().detach().numpy())    
            pred.extend(torch.argmax(output,dim =1).cpu().detach().numpy())
            pred_l.extend(output[:,1].cpu().detach().numpy())
            true.extend(label.cpu().detach().numpy())
                         
    return pred, pred_l, true, running_loss, c


if __name__ == '__main__':
    
    
    import optuna
   
    
    batch_size = 16
    
    # model = get_resnet_lrelu(10,3,0,[8, 8, 16, 16],0.1,[0,0,0],[32],0,[16],[0, 0, 0])
   
    loss= nn.CrossEntropyLoss(label_smoothing=0)
 
    epochs = 120
    sm = nn.Softmax(dim = 1)
    
    d = {'Parameters': ['train_auc','val_auc','train_loss','val_loss','epochs','train_acc','vall_acc','test_loss','test_acc','test_auc'],
         'bsl':[],'w1':[],'w2':[],'w3':[],'w4':[],'w5':[],'w6':[],'w7':[],
         '5w':[],'6M':[],'12M':[],'18M':[],'24M':[],'36M':[],'48M':[],'60M':[]}
    time_keys = ['bsl','w1','w2','w3','w4','w5','w6','w7','5w','6M','12M','18M','24M','36M','48M','60M',]

    
    for time_point_idx in range(10):
        
        def objective(trial):
            
            t_l, v_l, t_c, v_c, t_auc, v_auc = [],[],[],[],[],[]
            
            # Model parameters
            out_path = r'C:\Users\Luuk\Desktop\Optuna/' + time_keys[time_point_idx] + '/'       
            # set parameters
            n_input_channels = 3
            depth = 96
            height = 96
            width = 96
            n_features = 0
            num_classes = 2
            pad_value = 0 
            lrelu_alpha = 0.1
            # To be varied parameters
            n_down_blocks = trial.suggest_int("num_blocks",2,5,step = 1)
            
            filters = []
            kernel_sizes = []
            strides = []
            
            for i in range(n_down_blocks):
                
                # try:
                #    n_units = trial.suggest_int("num_units" + str(i),filters[-1],64,step = 8)
                # except:    
                n_units = trial.suggest_int("num_units" + str(i),8,64,step = 8)
                filters.append(n_units)
                kernel_size = trial.suggest_int("kernel_size" + str(i),3,5,step = 2)
                kernel_sizes.append(kernel_size)
                strides.append([1])
                
            # filters = [8, 8, 16, 16, 32]
            # kernel_sizes = [7, 5, 4, 3, 3]  
            
            n_lin_layers = trial.suggest_int("lin_layers",2,5,step = 1)
            
            linear_units = []
            dropout_p = []
            
            for i in range(n_lin_layers-1):
                
                # try:
                #    lin_units = trial.suggest_int("num_units" + str(i),linear_units[-1],64,step = 8)
                # except:    
                lin_units = trial.suggest_int("num_lin_units" + str(i),8,64,step = 8)
                
                # lin_units = trial.suggest_int("num_units_lin" + str(i),8,64,step = 8)
                linear_units.append(lin_units)
                dropout_p.append(0)
    
            
            # linear_units = [16]          
            # dropout_p = [0]
            # To be varied experiments
            Impute = trial.suggest_categorical('imp',[True,False])
            resampling_meth = trial.suggest_categorical("Re_sample",["None", "up", "down"])
            
            model = DCNN_Pooling(n_input_channels = n_input_channels,
            n_features = n_features,
            filters = filters,
            kernel_sizes = kernel_sizes,
            pad_value = pad_value,
            n_down_blocks = len(filters),
            lrelu_alpha = lrelu_alpha,
            linear_units = linear_units, strides = strides)
            
            model.to('cuda:1')
            
            lr = trial.suggest_float('learning_rate',1e-5,1e-2,log = True)
            
            optimizer = torch.optim.Adam(model.parameters(),lr)
            
            epochs = 120
            
            for i in range(epochs):
                train,validation,test = get_data_loader(batch_size,imputed=Impute,
                                                        time_point_idx = time_point_idx, label_eq = True, label_eq_meth = resampling_meth)
                running_loss = 0
                c = 0
                pred = []
                true = []
                
                pred_l = []
                model.train()
                for a,b in enumerate(train):
                    inp = b[0].requires_grad_().cuda('cuda:1')
                    label = b[1].type(torch.LongTensor).cuda('cuda:1')
                    optimizer.zero_grad()
                    output = model(inp,None)
                    output = sm(output)
                    ls = loss(output,label)
                    ls.backward()
                    optimizer.step()
                    running_loss += ls.cpu().detach().numpy()
                    c += np.sum((torch.argmax(output,dim =1) == label).cpu().detach().numpy())
                    pred.extend(torch.argmax(output,dim =1).cpu().detach().numpy())
                    pred_l.extend(output[:,1].cpu().detach().numpy())
                    true.extend(label.cpu().detach().numpy())
        
                    # scheduler.step(i+(a+1)/len(train))
                   
                pred_val, pred_l_val, true_val, running_loss_val, c_val = evaluate_model(model,validation)
                
                print(i,'Train',running_loss/(len(train)),c/(len(train)*batch_size),auc(true,pred_l),'| val',running_loss_val/(len(validation)),c_val/(len(validation)*batch_size),auc(true_val,pred_l_val))
                t_l.append(running_loss/(len(train)))
                v_l.append(running_loss_val/(len(validation)))
                t_c.append(c/(len(train)*batch_size),)
                v_c.append(c_val/(len(validation)*batch_size))
                t_auc.append(auc(true,pred_l))
                v_auc.append(auc(true_val,pred_l_val))
                # fpr, tpr, thresholds = roc(true_val, pred_l_val)
                if early_stopping(v_l):
                    break
    
    
            try:
                best_val = study.best_trial.value
            except:
                best_val = np.inf
    
            if running_loss_val/(len(validation)) <= best_val:
                
                best_path = out_path +'Best_in_TP' + str(time_point_idx)
                torch.save(model.state_dict(), best_path) 
                
                # Evaluate the best model on the testing data
                model.load_state_dict(torch.load(best_path))
                
                # save_predictions(model,time_point_idx, out_path + 'pred_info_' + str(time_point_idx)+ '.xlsx')
                
                pred_test, pred_l_test, true_test, running_loss_test, c_test = evaluate_model(model,test)
                pred_val, pred_l_val, true_val, running_loss_val, c_val = evaluate_model(model,validation)
                pred, pred_l, true, running_loss, c = evaluate_model(model,train)

                
                # out_path = out_path + '/' + time_keys[time_point_idx] + '/'
                
                # Visualizations
                loss_plot(t_l, v_l, i+1, out_path + 'loss.png')
                loss_plot(t_auc, v_auc, i+1, out_path + 'AUC.png')
                
                confusion_matrix_bin(true, pred, out_path + 'conf_train.png')
                confusion_matrix_bin(true_val, pred_val, out_path + 'conf_val.png')
                confusion_matrix_bin(true_test, pred_test, out_path + 'conf_test.png')
                
                calibration_plot(true,pred_l,5, out_path + 'cal_train.png')
                calibration_plot(true_val,pred_l_val,5, out_path + 'cal_val.png')
                calibration_plot(true_test,pred_l_test,5, out_path +  'cal_test.png') 
                
                save_predictions(model,time_point_idx,out_path + 'pred.xlsx')
                
            return running_loss_val/(len(validation))
        
        path = r'C:\Users\Luuk\Desktop\Results_DCNN_test\Studies/'
        study = optuna.create_study(direction='minimize')
        study.optimize(objective,n_trials = 50)
        joblib.dump(study, path + "study_" + time_keys[time_point_idx] + '.pkl')