import wandb
import optuna
import logging

from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset_total
from src.models.tools.save_model import save_model, save_config, save_dataset, save_dataset_summary
from src.training.train_multi import train, validate
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.utils.loss_func.get_loss_function import get_loss_function
from src.utils.fileHandler import create_file, create_folder, create_textfile
from src.dataset.get_dataloader import make_dataloader
from src.dataset.get_transforms import get_transforms

from torch.utils.data import DataLoader
import os
import numpy as np

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
        if(config['general']['testMode']):
            logging.info("WARNING: ------------ TEST MODE IS ACTIVE -------------")
        if(self.Optuna_study != None):
            self.Optuna_study.optimize(lambda trial: UpdateTrial(self, trial, config), config['hyperparam_tuning']['optuna']['n_trials'])
        else:
            UpdateTrial(self, None, config)

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
        create_file(pathOptunaStudyTracker)
        print(storage_name)

        study = optuna.create_study(study_name = studyName, storage = storage_name, load_if_exists=True)
        config['general']['firstRun'] = (len(study.get_trials()) == 0)
        if(config['general']['firstRun']):
            print("First optuna run has been detected!")
    return study

def UpdateTrial(hyperClass, trial, config):
    if(trial != None):
        # Alter normal hyperparameters
        config = normal_hyperparameters(trial,config)

        # Alter any hyperparameters that are dependend on other parameters 
        config = derived_hyperparameters(trial, config)

        if(config['hyperparam_tuning']['WandB']['IsEnabled']):
            # Setup WandB for run
            configWandB = dict(trial.params)  
            configWandB["trial.number"] = trial.number
        config['general']['trialNumber'] = f"Trial_{trial.number}" 
    else:
        subdirs = os.listdir(config['paths']['results'])
        numHigh = 0
        for i in range(len(subdirs)):
            valueFound = 0
            try:
                valueFound = int(subdirs[i][-3:])
            except:
                pass
            if(valueFound > numHigh):
                numHigh =  valueFound
        trialNumber =  numHigh + 1
        config['general']['trialNumber'] = f"Trial_{str(trialNumber).rjust(3,'0')}" 

    # Create result directory
    CreateResultDir(config)
    
    # Set loss function
    loss_function = get_loss_function(config)

    # Create the dataset dataframes
    DFs_col = load_dataset_total(config)
    trainDF_col = DFs_col[0]
    valDF_col= DFs_col[1]
    testDF_col = DFs_col[2]

    # make the training and validation transforms
    train_transforms, val_transforms = get_transforms(config)

    # Check if Kfolds is active --> group
    groupVar = None
    if(len(trainDF_col) > 1):
        groupVar = f"Folds{config['general']['trialNumber']}"


    logging.info(f"Dataset folds: {len(trainDF_col)}")
    # Amount of Ksplits
    val_loss_col = []
    val_auc_col = []
    for i in range(len(trainDF_col)):
        if(groupVar != None):
            CreateResultDir(config,i)

        if(trial != None and config['hyperparam_tuning']['WandB']['IsEnabled']):
            CreateStudy(config,configWandB,groupVar)

        train_loader, metadata = make_dataloader(config, trainDF_col[i], train_transforms, validation_mode=False)
        val_loader, _ = make_dataloader(config, valDF_col[i], val_transforms, validation_mode=True)

         # Get the data loaders
        # train_loader = DataLoader(trainDataset_col[i], batch_size=config['training']['batch_size'], shuffle=True, 
        #                               num_workers = config['data']['dataloader']['num_workers'], persistent_workers = config['data']['dataloader']['persistent_workers'])
        # val_loader = DataLoader(valDataset_col[i], batch_size=config['training']['batch_size'], shuffle=False,
        #                          num_workers = 1, persistent_workers = config['data']['dataloader']['persistent_workers'])
        try:
            model = train(config, train_loader, val_loader, metadata, hyperClass = hyperClass)
        except Exception as error:
            print(f"Warning: The training stopped due to an error. Please read the error message carefully: \n{str(error)}")
            create_textfile(config['general']['resultsCurrentDirectory'],"ErrorLog",str(error))
            if(config['general']['error_ignore']):
                return None
            else:
                raise Exception("Stopped trials due to error.")

        # Validation dataset check
        val_loss, val_auc = validate(loss_function, model, val_loader, config)
        val_loss_col.append(val_loss)
        val_auc_col.append(val_auc)

        # Test dataset check
        try:
            if(testDF_col[i].df.shape[0] != 0):
                test_loader, _ = make_dataloader(config, testDF_col[i], val_transforms, validation_mode=True)
                # test_loader = DataLoader(testDF_col[i], batch_size=config['training']['batch_size'], shuffle=False,
                #                          num_workers = 1, persistent_workers = config['data']['dataloader']['persistent_workers'])
                test_loss, test_auc = validate(loss_function, model, test_loader, config)
                savePath = os.path.join(config['general']['resultsCurrentDirectory'],"test_loss.txt")
                f = open(savePath,"w")
                f.close()
                np.savetxt(savePath,np.array(test_loss))
                savePath = os.path.join(config['general']['resultsCurrentDirectory'],"test_auc.txt")
                f = open(savePath,"w")
                f.close()
                np.savetxt(savePath,np.array(test_auc))
        except Exception as e:
            logging.info(f"Could not validate the test dataset with error: {e}")

        # Save model
        try:
            save_model(config, model, f"DlModel_Weights.pth")
            save_config(config,f"DlModel_Config.yaml")
        except:
            print("WARNING: SAVING NOT WORKING!! PLEASE CHECK")

        # Save Datasets
        save_dataset(config,trainDF_col[i],"train_Dataset")
        save_dataset(config,valDF_col[i],"validation_Dataset")
        save_dataset(config,testDF_col[i],"test_Dataset")
        save_dataset_summary(config,trainDF_col[i],valDF_col[i],testDF_col[i])

        logging.info("Information about the val_auc_array:")
        logging.info(val_auc_col)
        #if(np.mean(np.array(val_auc_col)) < 0.7 and i >= 1 and len(trainDataset_col) -1 != i):
        #    logging.info(f"Stopped the folds due to a low validation AUC.")
        #    break

        # Check if run needs to be aborted
        if(config['run']['patienceExhausted'] and (config['run']['patienceExhaustedIndex'] < (config['training']['patience']*2 + 2)) and i == 0):
            if(len(trainDF_col) > 1):
                logging.info("Patience exhausted and ignoring K-split datasets")
            break

    val_lossMean =  np.mean(val_loss_col)
    if(groupVar != None):
        val_lossMean =  np.mean(val_loss_col)
        logging.info(f"The mean loss of K-folds is: {round(val_lossMean,2)}")
    else:
        val_lossMean = val_loss_col[0]
        # Create a results file for the Folds results and the mean values    

    return val_lossMean

def update_config(dic, location, suggested_value):
    if len(location) == 1:
        dic[location[0]] = suggested_value
        return dic
    else:
        dic[location[0]] = update_config(dic[location[0]], location[1:], suggested_value)
        return dic
    

def check_names(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def CreateResultDir(config, KFoldIndex = -1):
    folderPath = os.path.join(os.path.join(config["paths"]["output"],config["hyperparam_tuning"]["ProjectName"]),config["general"]["trialNumber"])
    if(KFoldIndex != -1):
        folderPath = os.path.join(folderPath,f"KFold{KFoldIndex}")
    folderPath += "\\"

    # Create folder if does not exist
    create_folder(folderPath)
    check_names(folderPath)

    logging.info(f"Directory path updated: {folderPath}")
    config['general']['resultsCurrentDirectory'] = folderPath
