# -*- coding: utf-8 -*-

import optuna
from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset_single
from src.dataset.get_dataloader import make_dataloader
from src.dataset.get_transforms import get_transforms
from src.models.tools.save_model import save_model
from src.training.train import train, validate
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.utils.loss_func.get_loss_function import get_loss_function
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

def normal_hyperparameters(config, trial):
    for key in config['hyperparam_tuning']['hyperparams'].keys():
        try:
            hyperinfo = config['hyperparam_tuning']['hyperparams'][key]
            location = hyperinfo['location']

            # Check if multi is encountered
            if(hyperinfo['name'].startswith("n_")):
                suggested_value = generate_value(trial, hyperinfo, config['general']['firstRun'])
            else:   
                suggested_value = generate_value(trial, hyperinfo)
            
            config = update_config(config,location,suggested_value)
        except Exception as error:
                print(f"Could not generate hyperparams [Single] with the name '{key}' with following error:\n {error}")
                raise Exception("Stopped trials due to error.")
    return config

def derived_hyperparameters(config, trial):
    '''
    Some hyperparameters are connected/ interdependend, here derived 
    hyperparameters are to be set.
    
    '''
    # If the number of down blocks is specified there should be an equal amount of related parameters   
    if 'n_down_blocks' in config['hyperparam_tuning']['hyperparams'].keys():
        for key in config['hyperparam_tuning']['derived']['n_down_blocks']:
            try:
                temp_list = []
                hyperinfo = config['hyperparam_tuning']['derived']['n_down_blocks'][key]  
                location = hyperinfo['location']
                base_name = hyperinfo['name']
                for i in range(config['model']['n_down_blocks']):
                    hyperinfo_new = hyperinfo.copy()
                    hyperinfo_new['name'] = base_name + str(i)
                    suggested_value = generate_value(trial, hyperinfo_new)
                    temp_list.append(suggested_value)
                config = update_config(config,location,temp_list)
            except Exception as error:
                print(f"Could not generate hyperparams [Derived] with the name '{key}' with following error:\n {error}")
                raise Exception("Stopped trials due to error.")
            
    if 'n_clinical_down_blocks' in config['hyperparam_tuning']['hyperparams'].keys():
        for key in config['hyperparam_tuning']['derived']['n_clinical_down_blocks']:
            try:
                temp_list = []
                hyperinfo = config['hyperparam_tuning']['derived']['n_clinical_down_blocks'][key]  
                location = hyperinfo['location']
                base_name = hyperinfo['name']

                if(base_name == "clin_pos"):
                    hyperinfo['max'] = config['model']['n_clinical_down_blocks']
                    suggested_value = generate_value(trial, hyperinfo)
                    config = update_config(config,location,suggested_value)
                else:
                    for i in range(config['model']['n_clinical_down_blocks']):
                        hyperinfo_new = hyperinfo.copy()
                        hyperinfo_new['name'] = base_name + str(i)
                        suggested_value = generate_value(trial, hyperinfo_new)
                        temp_list.append(suggested_value)
                    config = update_config(config,location,temp_list)

            except Exception as error:
                print(f"Could not generate hyperparams [Derived] with the name '{key}' with following error:\n {error}")
                raise Exception("Stopped trials due to error.")
            
    if 'n_linear_down_blocks' in config['hyperparam_tuning']['hyperparams'].keys():
        for key in config['hyperparam_tuning']['derived']['n_linear_down_blocks']:
            try:
                temp_list = []
                hyperinfo = config['hyperparam_tuning']['derived']['n_linear_down_blocks'][key]  
                location = hyperinfo['location']
                base_name = hyperinfo['name']
                for i in range(config['model']['n_linear_down_blocks']):
                    hyperinfo_new = hyperinfo.copy()
                    hyperinfo_new['name'] = base_name + str(i)
                    suggested_value = generate_value(trial, hyperinfo_new)
                    temp_list.append(suggested_value)
                config = update_config(config,location,temp_list)
            except Exception as error:
                print(f"Could not generate hyperparams [Derived] with the name '{key}' with following error:\n {error}")
                raise Exception("Stopped trials due to error.")
            
    if 'n_linear_units_endpoint' in config['hyperparam_tuning']['hyperparams'].keys():
        for key in config['hyperparam_tuning']['derived']['n_linear_units_endpoint']:
            try:
                temp_list = []
                hyperinfo = config['hyperparam_tuning']['derived']['n_linear_units_endpoint'][key]  
                location = hyperinfo['location']
                base_name = hyperinfo['name']
                for i in range(config['model']['n_linear_units_endpoint']):
                    hyperinfo_new = hyperinfo.copy()
                    hyperinfo_new['name'] = base_name + str(i)
                    suggested_value = generate_value(trial, hyperinfo_new)
                    temp_list.append(suggested_value)
                config = update_config(config,location,temp_list)
            except Exception as error:
                print(f"Could not generate hyperparams [Derived] with the name '{key}' with following error:\n {error}")
                raise Exception("Stopped trials due to error.")
        
    return config

def set_minValue(hyperinfo, firstRun):
    minValue = hyperinfo['min']
    if(firstRun):
        minValue = hyperinfo['max']
    return minValue


def generate_value(trial, hyperinfo, firstRun = False):
    """
    Selects and generates a value based on hyperinfo
    """
    keys = hyperinfo.keys()
    step = None
    log = False
    if('step' in keys):
        step = hyperinfo['step']
    if('log' in keys):
        log = hyperinfo['log']

    if hyperinfo['type'] == 'int':
        if(step == None):
            step = 1
        suggested_value = trial.suggest_int(hyperinfo['name'], set_minValue(hyperinfo,firstRun), hyperinfo['max'],step=step,log=log)
    if hyperinfo['type'] == 'float':
        suggested_value = trial.suggest_float(hyperinfo['name'],set_minValue(hyperinfo,firstRun),hyperinfo['max'],step=step,log=log)
    if hyperinfo['type'] == 'categorical':
        suggested_value = trial.suggest_categorical(hyperinfo['name'],hyperinfo['options'])       
    if hyperinfo['type'] == 'dis_un':
        suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'],set_minValue(hyperinfo,firstRun),hyperinfo['max'],hyperinfo['q'])       
    if hyperinfo['type'] == 'log_un':
        suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'],set_minValue(hyperinfo,firstRun),hyperinfo['max'])
    if hyperinfo['type'] == 'un':
        suggested_value = trial.suggest_discrete_uniform(hyperinfo['name'],set_minValue(hyperinfo,firstRun),hyperinfo['max'])       
    return suggested_value

# def save_trial(model, config, trial_num):
#     out_path = config['hyperparam_tuning']['optuna']['trial_results']
        
#     if not os.path.exists(out_path):
#         os.makedirs(out_path)
    
#     return




# def Optuna_CreateStudy(config):
#     study = None
#     if(config['hyperparam_tuning']['optuna']['IsEnabled']):

#         # Check where to keep study information
#         studyName = config['general']['experiment_name']
#         pathOptunaStudyTracker = os.path.join(os.path.join(config['paths']['results'] , studyName), "track_optuna.db")
#         storage_name = f"sqlite:///{pathOptunaStudyTracker}"
#         create_folder(pathOptunaStudyTracker)
#         print(storage_name)

#         study = optuna.create_study(study_name = studyName, storage = storage_name, load_if_exists=True, direction = config['hyperparam_tuning']['optuna']['direction'])
#     return study



def Optuna_initialise_study(config):
    study = None
    if (config['hyperparam_tuning']['optuna']['IsEnabled']):

        # Check where to keep study information
        studyName = config['general']['experiment_name']
        pathOptunaStudyTracker = os.path.join(os.path.join(config['paths']['results'] , studyName), "track_optuna.db")
        storage_name = f"sqlite:///{pathOptunaStudyTracker}"
        create_file(pathOptunaStudyTracker)
        print(storage_name)

        study = optuna.create_study(
                            study_name = studyName, 
                            storage = storage_name, 
                            load_if_exists=True,
                            directions= config['hyperparam_tuning']['optuna']['objective_direction']
                            )

        # check if an experiment has been run before
        config['general']['firstRun'] = (len(study.get_trials()) == 0)
        if (config['general']['firstRun']):
            print("First optuna run has been detected!")

    return study



