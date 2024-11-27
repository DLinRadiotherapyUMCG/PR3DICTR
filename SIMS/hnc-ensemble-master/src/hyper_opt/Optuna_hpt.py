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
from src.utils.fileHandler import create_file, create_folder

from torch.utils.data import DataLoader
import os

from functools import partial

def update_config(dic, location, suggested_value):
    """
    Function to update the config based on a given location and 
    """
    if len(location) == 1:
        dic[location[0]] = suggested_value
        return dic
    else:
        dic[location[0]] = update_config(dic[location[0]], location[1:], suggested_value)
        return dic

def derived_hyperparameters(trial, config):
    '''
    Some hyperparameters are connected/ interdependend, here derived 
    hyperparameters are to be set.
    
    '''
    # If the number of down blocks is specified there should be an equal amount of related parameters   
    if 'n_down_blocks' in config['hyperparam_tuning']['hyperparams'].keys():
        for key in config['hyperparam_tuning']['derived']['n_down_blocks']:
            temp_list = []
            hyperinfo = config['hyperparam_tuning']['hyperparams'][key]  
            location = hyperinfo['location']
            base_name = hyperinfo['name']
            for i in range(config['model']['n_down_blocks']):
                hyperinfo['name'] = base_name + str(i)
                suggested_value = generate_value(trial, hyperinfo)
                temp_list.append(suggested_value)
            config = update_config(config,location,temp_list)
            
    return config


def generate_value(trial, hyperinfo):
    """
    Selects and generates a value based on hyperinfo
    """
    if hyperinfo['type'] == 'int':
        suggested_value = trial.suggest_int(hyperinfo['name'],hyperinfo['min'],hyperinfo['max'],hyperinfo['step'],hyperinfo['log'])
    if hyperinfo['type'] == 'float':
        suggested_value = trial.suggest_float(hyperinfo['name'],hyperinfo['min'],hyperinfo['max'],hyperinfo['step'],hyperinfo['log'])
    if hyperinfo['type'] == 'categorical':
        suggested_value = trial.suggest_categorical(hyperinfo['name'],hyperinfo['options'])       
    if hyperinfo['type'] == 'dis_un':
        suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'],hyperinfo['min'],hyperinfo['max'],hyperinfo['q'])       
    if hyperinfo['type'] == 'log_un':
        suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'],hyperinfo['min'],hyperinfo['max'])
    if hyperinfo['type'] == 'un':
        suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'],hyperinfo['min'],hyperinfo['max'])       
    return suggested_value

def save_trial(model, config, trial_num):
    out_path = config['hyperparam_tuning']['optuna']['trial_results']
        
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    return

def objective(trial, config):
    """
    1) Create a "working" config file 
    2) Use this to train a model
    3) evaluate and return model performance

    """
    
    trial_num = trial.number
    config['hyperparam_tuning']['optuna']['trial_results'] = config['hyperparam_tuning']['ProjectName'] + '_' + trial_num + '/'
    
    
    # Assign values for "regular" hyperparameters
    for key in config['hyperparam_tuning']['hyperparams'].keys():
        hyperinfo = config['hyperparam_tuning']['hyperparams'][key]
        location = hyperinfo['location']
        suggested_value = generate_value(trial, hyperinfo)
        config = update_config(config,location,suggested_value)

    # Alter any hyperparameters that are dependend on other parameters 
    config = derived_hyperparameters(trial, config)

    # Train with the local config file
    loss_function = get_loss_function(config)

    # load data
    train_data, metadata = load_dataset(config, os.path.join(config['paths']['csv'], config['filenames']['train_csv']), augment=True)
    val_data, _ = load_dataset(config, os.path.join(config['paths']['csv'], config['filenames']['validation_csv']))
    val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Train model 
    model = train(config, train_data, val_data, metadata)
    
    save_trial(model, config)
    
    # Generate validation results
    val_loss, val_auc = validate(loss_function, model, val_loader)
    
    return val_loss

def Optuna_CreateStudy(config):
    study = None
    if(config['hyperparam_tuning']['optuna']['IsEnabled']):

        # Check where to keep study information
        studyName = config['hyperparam_tuning']['ProjectName']
        pathOptunaStudyTracker = os.path.join(os.path.join(config['paths']['results'] , studyName), "track_optuna.db")
        storage_name = f"sqlite:///{pathOptunaStudyTracker}"
        create_folder(pathOptunaStudyTracker)
        print(storage_name)

        study = optuna.create_study(study_name = studyName, storage = storage_name, load_if_exists=True, direction = config['hyperparam_tuning']['optuna']['direction'])
    return study

def Optuna_optimalisation(config):
    """
    
    """
    study = Optuna_CreateStudy(config)
    
    study.optimize(lambda trial: objective(trial, config), config['hyperparam_tuning']['optuna']['n_trails'])
    
    return