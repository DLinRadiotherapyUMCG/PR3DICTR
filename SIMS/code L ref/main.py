# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:53:31 2024

@author: HoekL02
"""
from torch.utils.data import DataLoader
from resnet_relu import get_resnet_lrelu
from data_preparation import get_data_loader
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import roc_curve as roc

import numpy as np

from methods import DCNN_LReLU, AdaBound

import pandas as pd

from DCNN_pooling import DCNN_Pooling

from visualization import loss_plot, confusion_matrix_bin, calibration_plot

def evaluate_model(model,data):
    
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






if __name__ == '__main__':
    
    
    
    batch_size = 8
   
    loss= nn.CrossEntropyLoss(label_smoothing=0)
 
    epochs = 100
    sm = nn.Softmax(dim = 1)
    
    d = {'Parameters': ['train_auc','val_auc','train_loss','val_loss','epochs','train_acc','vall_acc','test_loss','test_acc','test_auc'],
         'bsl':[],'w1':[],'w2':[],'w3':[],'w4':[],'w5':[],'w6':[],'w7':[],
         '5w':[],'6M':[],'12M':[],'18M':[],'24M':[],'36M':[],'48M':[],'60M':[]}
    time_keys = ['bsl','w1','w2','w3','w4','w5','w6','w7','5w','6M','12M','18M','24M','36M','48M','60M',]
    
    data_save = {'tp': [], 'train_auc': [], 'val_auc': [], 'train_acc': [], 'val_acc': []}
    data_save_loss = {'tp': [], 'train_auc': [], 'val_auc': [], 'train_acc': [], 'val_acc': []}
    
    for time_point_idx in range(16):   
        out_path = r'C:\Users\Luuk\Desktop\Results_DCNN_test/'
        
        
        t_l, v_l, t_c, v_c, t_auc, v_auc = [],[],[],[],[],[]
        
        model = DCNN_Pooling(n_input_channels = 3,
        n_features = 0,
        filters = [8, 8, 16, 16, 32],
        kernel_sizes = [7, 5, 4, 3, 3],
        strides = [[1]*3, [1]*3, [1]*3, [1]*3, [1]*3],
        pad_value = 0,
        lrelu_alpha = 0.1, pooling = 'max')
        
        model.to('cuda')
        
        # optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        optimizer = AdaBound(model.parameters(), lr=1e-4, final_lr=1e-4 * 100, weight_decay=0.05)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=40,
                                                                     T_mult=1,
                                                                     eta_min=1e-8)
        
        for i in range(epochs):
            train,validation,test = get_data_loader(batch_size,imputed=True, time_point_idx = time_point_idx)
            running_loss = 0
            c = 0
            pred = []
            true = []
            
            pred_l = []
            model.train()
            for a,b in enumerate(train):
                inp = b[0].requires_grad_().cuda()
                label = b[1].type(torch.LongTensor).cuda()
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
    
                scheduler.step(i+(a+1)/len(train))
               
            pred_val, pred_l_val, true_val, running_loss_val, c_val = evaluate_model(model,validation)
            
            print(i,'Train',running_loss/(len(train)),c/(len(train)*batch_size),auc(true,pred_l),'| val',running_loss_val/(len(validation)),c_val/(len(validation)*batch_size),auc(true_val,pred_l_val))
            t_l.append(running_loss/(len(train)))
            v_l.append(running_loss_val/(len(validation)))
            t_c.append(c/(len(train)*batch_size),)
            v_c.append(c_val/(len(validation)*batch_size))
            t_auc.append(auc(true,pred_l))
            v_auc.append(auc(true_val,pred_l_val))
            # fpr, tpr, thresholds = roc(true_val, pred_l_val)
            
            if running_loss_val/(len(validation)) <= np.min(v_l):

                best_path = out_path +'Best_in_TP' + str(time_point_idx)
      
                torch.save(model.state_dict(), out_path +'Best_in_TP' + str(time_point_idx)) 

      
        # Evaluate the best model on the testing data
        model.load_state_dict(torch.load(best_path))
        pred_test, pred_l_test, true_test, running_loss_test, c_test = evaluate_model(model,test)
        pred_val, pred_l_val, true_val, running_loss_val, c_val = evaluate_model(model,validation)
        pred, pred_l, true, running_loss, c = evaluate_model(model,train)
        
        
        data_save['tp'].append(time_keys[time_point_idx])
        data_save['val_auc'].append(v_auc[np.argmax(v_auc)])
        data_save['train_auc'].append(t_auc[np.argmax(v_auc)])
        data_save['train_acc'].append(t_c[np.argmax(v_auc)])
        data_save['val_acc'].append(v_c[np.argmax(v_auc)])
        df = pd.DataFrame(data_save)
        df.to_excel(out_path + '/' + 'results_gen.xlsx')
        
        data_save_loss['tp'].append(time_keys[time_point_idx])
        data_save_loss['val_auc'].append(v_auc[np.argmin(v_l)])
        data_save_loss['train_auc'].append(t_auc[np.argmin(v_l)])
        data_save_loss['train_acc'].append(t_c[np.argmin(v_l)])
        data_save_loss['val_acc'].append(v_c[np.argmin(v_l)])
        df_loss = pd.DataFrame(data_save_loss)
        df_loss.to_excel(out_path + '/' + 'results_gen_loss.xlsx')   
        
        
        # below save df info 'train_auc','val_auc','train_loss','val_loss','epochs','train_acc','vall_acc','test_loss','test_acc','test_auc'
        # metrics = [t_auc[epoch_c],v_auc[epoch_c],t_l[epoch_c],v_l[epoch_c],epoch_c,t_c[epoch_c],v_c[epoch_c],t_ECE[epoch_c],v_ECE[epoch_c],
        #            t_F1[epoch_c],v_F1[epoch_c],
        #            running_loss_test/len(test),auc(true_test,pred_l_test),
        #            ECE(torch.tensor(true_test),torch.tensor(pred_l_test)),F1(torch.tensor(true_test),torch.tensor(pred_test)),c_test/(len(test)*batch_size)]
        
        
        out_path = out_path + '/' + time_keys[time_point_idx] + '/'
        

        
        # Visualizations
        loss_plot(t_l, v_l, epochs, out_path + 'loss.png')
        loss_plot(t_auc, v_auc, epochs, out_path + 'AUC.png')
        
        confusion_matrix_bin(true, pred, out_path + 'conf_train.png')
        confusion_matrix_bin(true_val, pred_val, out_path + 'conf_val.png')
        confusion_matrix_bin(true_test, pred_test, out_path + 'conf_test.png')
        
        calibration_plot(true,pred_l,5, out_path + 'cal_train.png')
        calibration_plot(true_val,pred_l_val,5, out_path + 'cal_val.png')
        calibration_plot(true_test,pred_l_test,5, out_path +  'cal_test.png')
    