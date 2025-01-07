import logging
import os
import src
import wandb

from src.constants import DEVICE

from src.config_presets.tools.get_config import get_config
from src.config_presets.tools.load_config import load_config
from src.dataset.load_dataset import load_dataset, load_dataset_total
from src.models.tools.save_model import save_model
from src.training.train import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.hyper_opt.hyperHandler import HyperTuning_Handler
from src.utils.fileHandler import create_file
from sklearn.metrics import roc_auc_score
from src.utils.move_batch_to_device import move_batch_to_device

from src.models.tools.get_classification_model import get_classification_model

from torch.utils.data import DataLoader

import numpy as np 

import torch

import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

# def move_batch_to_device(batch, device):         # NOTE: Daniel has replaced this with an imported function from src.training.utils 
#     inputs, clinical_features, targets = batch
#     inputs = inputs.to(device=device)
#     clinical_features = clinical_features.to(device=device)
#     targets = targets.to(device=device)
#     return inputs, clinical_features, targets

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


def load_CITOR_features(config,relevant_columns, train_data, val_data, toxicity):
    label = config['CITOR'][toxicity]['label']
    X_train = train_data[relevant_columns][train_data[label] != -1]
    y_train = train_data[label][train_data[label] != -1]
    X_val = val_data[relevant_columns][val_data[label] != -1]
    y_val = val_data[label][val_data[label] != -1]    
    return X_train, y_train, X_val, y_val

def CITOR_refit(config, toxicity = 'sticky_saliva_late'):
    
    train_data, val_data = lr_load_data(config)
    
    if len(config['CITOR'][toxicity]['models'].keys()) >1:
        feature_dict = {'model1':{},'model2':{}}
        model_names = ['model1','model2']
    else:
        feature_dict = {'model1':{}}
        model_names = ['model1']     

    for model_name in model_names:
       # select the columns per submodel
        relevant_columns = np.array(config['CITOR'][toxicity]['models'][model_name]['features'])
        # train the submodel
        X_train, y_train, X_val, y_val = load_CITOR_features(config,relevant_columns, train_data, val_data, toxicity)
        model, auc = lr_fit_model(config, X_train, y_train, X_val, y_val)

        # save the results from the submodel
        feature_dict[model_name]['features'] = relevant_columns
        feature_dict[model_name]['coef'] = model.coef_[0]
        feature_dict[model_name]['intercept'] = model.intercept_[0]
    
    if len(feature_dict.keys()) > 1:
        model1 = list(feature_dict.keys())[0]
        model2 = list(feature_dict.keys())[1]
        features = []
        coef = []
        features.append('intercept')
        coef.append(.5*(feature_dict[model1]['intercept']+feature_dict[model2]['intercept']))
        
        features1 = feature_dict[model1]['features']
        features2 = feature_dict[model2]['features']
        
        overlap =  list(set(features1) & set(features2))
        remainder_m1 = list(set(features1) - set(features2))
        remainder_m2 = list(set(features2) - set(features1))

        for i in range(len(overlap)):
            features.append(overlap[i])
            coef.append(.5*(feature_dict[model1]['coef'][np.where(features1== overlap[i])]+feature_dict[model2]['coef'][np.where(features2 == overlap[i])])[0])
        for i in range(len(remainder_m1)):
            features.append(remainder_m1[i])
            coef.append(.5*(feature_dict[model1]['coef'][np.where(features1 == remainder_m1[i])])[0])         
        for i in range(len(remainder_m2)):
            features.append(remainder_m2[i])
            coef.append(.5*(feature_dict[model2]['coef'][np.where(features2== remainder_m2[i])])[0])         
   
    else:
        features =['intercept'] +  list(relevant_columns)
        coef =[model.intercept_[0]]+ list(model.coef_[0]) 
    
    label = config['CITOR'][toxicity]['label']
    intercept = coef[0]
    X = train_data[features[1:]][train_data[label] !=-1].to_numpy()
    y = train_data[label][train_data[label] != -1]
    
    pred = 1/(1+np.exp(-(intercept + np.dot(X,coef[1:]))))
    
    auc = roc_auc_score(y, pred)
    print('auc:',auc)
    
    return features, coef

if __name__ == '__main__':
    # Setup
    toxicity, log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = get_config('Multi_time')

    # Disable randomness
    set_random_seed(config['general']['seed'])

    # Test a chosen project
    nameProjectTest = "Resnet_depth_optimization_modeltype2"
    config['general']['experiment_name'] = nameProjectTest
    #config['hyperparam_tuning']['optuna']['studyname'] = nameProjectTest

    #Load the existing project
    hyperClass = HyperTuning_Handler(config)
    df = hyperClass.Optuna_study.trials_dataframe()

    # Show the project information
    print(df)

    # Close the project envrionment
    hyperClass.Stop()
    
    
    best = 4 #26
    config['paths']['results'] = "//zkh/appdata/RTDicom/Projectline_HNC_modelling/Users/Luuk vd Hoek/Projects collection/Mini projects/SIMS/hnc-ensemble-master/results/"
    
    path_results = os.path.join(config['paths']['results'],nameProjectTest, 'Trial_' + str(best))
    data, metadata = load_dataset_total(config)
    test_data = DataLoader(data[2][0],batch_size=config['training']['batch_size'], shuffle=False, 
                                      num_workers = config['data']['dataloader']['num_workers'], persistent_workers = config['data']['dataloader']['persistent_workers'])
    
    
    out_lists = []
    tar_lists  = []

    folds = 5
    
    for i in range(folds):
        path_fold = os.path.join(path_results,'Kfold' + str(i))
        config = load_config(path_fold + '/DlModel_Config.yaml')


        model = get_classification_model(config, metadata)
        model.cuda()
        model.load_state_dict(torch.load(path_fold + '/DlModel_Weights.pth'))
    
        out, targets = get_output_results(model, test_data, config)
        
    
        out_lists.append(out)
        tar_lists.append(targets)
        if i == 0:
            out_tot = dict.fromkeys(out.keys())
            for key in out_tot.keys():
                out_tot[key] = np.array(out[key])*(1/folds)
        else:
            for key in out_tot.keys():
                out_tot[key] = out_tot[key] + np.array(out[key])*(1/folds)
        
    for i in out_tot.keys():
        print(i,roc_auc_score(np.array(targets[i])[np.where(np.array(targets[i]) != -1)],np.array(out_tot[i])[np.where(np.array(targets[i]) != -1)]))