import wandb
import optuna
from src.get_config import get_config
from src.load_dataset import load_dataset, load_dataset_total
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

from src.hyper_opt.Optuna_hpt import *
from src.hyper_opt.WandB_hpt import *

class HyperTuning_Handler():
    def __init__(self,config):
        # Set optuna
        self.Optuna_study = Optuna_CreateStudy(config) 
        self.config = config

        # Set weight and biases
        WandB_initalise(config)

    def Operate(self, config):
        self.Optuna_study.optimize(lambda trial: UpdateTrial(self, trial, config), config['hyperparam_tuning']['optuna']['n_trails'])

    def UpdateWandB(self,results,epoch):
        UpdateStudy(self.config,results,epoch)

    def Stop(self):
        WandB_stop(self.config)

def Optuna_CreateStudy(config):
    study = None
    if(config['hyperparam_tuning']['optuna']['IsEnabled']):

        # Check where to keep study information
        studyName = config['hyperparam_tuning']['ProjectName']
        pathOptunaStudyTracker = os.path.join(os.path.join(config['paths']['results'] , studyName), "track_optuna.db")
        storage_name = f"sqlite:///{pathOptunaStudyTracker}"
        create_folder(pathOptunaStudyTracker)
        print(storage_name)

        study = optuna.create_study(study_name = studyName, storage = storage_name, load_if_exists=True)
    return study

def UpdateTrial(hyperClass, trial, config):
    for key in config['hyperparam_tuning']['hyperparams'].keys():
        hyperinfo = config['hyperparam_tuning']['hyperparams'][key]
        location = hyperinfo['location']
        if hyperinfo['type'] == 'int':
            suggested_value = trial.suggest_int(key,hyperinfo['min'],hyperinfo['max'])
            config = update_config(config,location,suggested_value)
        if hyperinfo['type'] == 'float':
            suggested_value = trial.suggest_float(key,hyperinfo['min'],hyperinfo['max'])
            config = update_config(config,location,suggested_value)
        if hyperinfo['type'] == 'categorical':
            suggested_value = trial.suggest_categorical(key,hyperinfo['options'])
            config = update_config(config,location,suggested_value)            
        if hyperinfo['type'] == 'dis_un':
            suggested_value = trial.suggest_discrete_uniform(key,hyperinfo['min'],hyperinfo['max'],hyperinfo['q'])
            config = update_config(config,location,suggested_value)          
        if hyperinfo['type'] == 'log_un':
            suggested_value = trial.suggest_discrete_uniform(key,hyperinfo['min'],hyperinfo['max'])
            config = update_config(config,location,suggested_value)   
        if hyperinfo['type'] == 'un':
            suggested_value = trial.suggest_discrete_uniform(key,hyperinfo['min'],hyperinfo['max'])
            config = update_config(config,location,suggested_value)

    # Setup WandB for run
    configWandB = dict(trial.params)  
    configWandB["trial.number"] = trial.number
    CreateStudy(config,configWandB)
    loss_function = get_loss_function(config)

    split = True
    set_random_seed(config['seed'])
    if(split == False):
        train_data, metadata = load_dataset(config, os.path.join(config['paths']['csv'], config['data']['trainfile']), augment=True, split = True, train = True, splitVar = config['data']['splitvar'])
        val_data, _ = load_dataset(config, os.path.join(config['paths']['csv'], config['data']['valfile']), split = True, train = False, splitVar = config['data']['splitvar'])
        val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'], shuffle=False)
    else:
        datasets, metadata = load_dataset_total(config)
        train_data = datasets[0]
        val_data = datasets[1]
        val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'], shuffle=False)

    try:
        model = train(config, train_data, val_data, metadata, hyperClass = hyperClass)
    except Exception as error:
        print(f"Warning: The training stopped due to an error. Please read the error message carefully: \n{error}")

    val_loss, val_auc = validate(loss_function, model, val_loader, config)

    save_model(config, model, 'dl_model_full.pth')

    return val_loss

def update_config(dic, location, suggested_value):
    if len(location) == 1:
        dic[location[0]] = suggested_value
        return dic
    else:
        dic[location[0]] = update_config(dic[location[0]], location[1:], suggested_value)
        return dic