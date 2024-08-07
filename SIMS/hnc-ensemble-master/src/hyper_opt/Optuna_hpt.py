# -*- coding: utf-8 -*-

import optuna
from src.get_config import get_config
from src.load_dataset import load_dataset
from src.save_model import save_model
from src.train_multi import train
from src.utils.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.train_multi import validate
from src.get_loss_function import get_loss_function

from torch.utils.data import DataLoader
import os

from functools import partial

def update_config(dic, location, suggested_value):
    if len(location) == 1:
        dic[location[0]] = suggested_value
        return dic
    else:
        dic[location[0]] = update_config(dic[location[0]], location[1:], suggested_value)
        return dic

def objective(trial, config):
    
    for key in config['hyperparam_tuning']['hyperparams'].keys():
        hyperinfo = config['hyperparam_tuning']['hyperparams'][key]
        location = hyperinfo['location']
        if hyperinfo['type'] == 'int':
            suggested_value = trial.suggest_int(hyperinfo['name'],hyperinfo['min'],hyperinfo['max'])
            config = update_config(config,location,suggested_value)
        if hyperinfo['type'] == 'float':
            suggested_value = trial.suggest_float(hyperinfo['name'],hyperinfo['min'],hyperinfo['max'])
            config = update_config(config,location,suggested_value)
        if hyperinfo['type'] == 'categorical':
            suggested_value = trial.suggest_categorical(hyperinfo['name'],hyperinfo['options'])
            config = update_config(config,location,suggested_value)            
        if hyperinfo['type'] == 'dis_un':
            suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'],hyperinfo['min'],hyperinfo['max'],hyperinfo['q'])
            config = update_config(config,location,suggested_value)          
        if hyperinfo['type'] == 'log_un':
            suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'],hyperinfo['min'],hyperinfo['max'])
            config = update_config(config,location,suggested_value)   
        if hyperinfo['type'] == 'un':
            suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'],hyperinfo['min'],hyperinfo['max'])
            config = update_config(config,location,suggested_value)   
            
    loss_function = get_loss_function(config)
    
    train_data, metadata = load_dataset(config, os.path.join(config['paths']['csv'], 'train_full.csv'), augment=True)
    val_data, _ = load_dataset(config, os.path.join(config['paths']['csv'], 'valid.csv'))
    val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'], shuffle=False)
    
    model = train(config, train_data, val_data, metadata)
    
    val_loss, val_auc = validate(loss_function, model, val_loader)
    
    return val_loss

def Optuna_optimalisation(config):
    
    study = optuna.create_study()
    
    study.optimize(lambda trial: objective(trial, config), config['hyperparam_tuning']['optuna']['n_trails'])
    
    return