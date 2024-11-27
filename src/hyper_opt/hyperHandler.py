import wandb
import optuna
import logging

from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset, load_dataset_total
from src.models.tools.save_model import save_model, save_config, save_dataset, save_dataset_summary
from src.training.train_multi import train, validate, validate_event
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.training.train_multi import move_batch_to_device, validate, get_output_results
from src.utils.loss_func.get_loss_function import get_loss_function
from src.utils.fileHandler import create_file, create_folder, create_textfile
from src.visualization.Confusion_matrix import confusion_matrix_multi

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
            if(config['training']['loss']['name'] != "NLLL"):
                print("------------------------------------------------------------------")
                print("-------------------- Start DL Binary Model---------------------")
                self.Optuna_study.optimize(lambda trial: UpdateBinaryTrial(self, trial, config), config['hyperparam_tuning']['optuna']['n_trials'])
            else: 
                print("------------------------------------------------------------------")
                print("-------------------- Start DL Event Time Model---------------------")
                self.Optuna_study.optimize(lambda trial: UpdateEventTrial(self, trial, config), config['hyperparam_tuning']['optuna']['n_trials'])
        else:
            UpdateBinaryTrial(self, None, config)

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

def UpdateBinaryTrial(hyperClass, trial, config):
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

    # Create the datasets
    datasets_col, metadata = load_dataset_total(config)
    trainDataset_col = datasets_col[0]
    valDataset_col= datasets_col[1]
    testDataset_col = datasets_col[2]

    # Check if Kfolds is active --> group
    groupVar = None
    if(len(trainDataset_col) > 1):
        groupVar = f"Folds{config['general']['trialNumber']}"


    logging.info(f"Dataset folds: {len(trainDataset_col)}")
    # Amount of Ksplits
    val_loss_col = []
    val_auc_col = []
    for i in range(len(trainDataset_col)):
        if(groupVar != None):
            CreateResultDir(config,i)

        if(trial != None and config['hyperparam_tuning']['WandB']['IsEnabled']):
            CreateStudy(config,configWandB,groupVar)

         # Get the data loaders
        train_loader = DataLoader(trainDataset_col[i], batch_size=config['training']['batch_size'], shuffle=True, 
                                      num_workers = config['data']['dataloader']['num_workers'], persistent_workers = config['data']['dataloader']['persistent_workers'])
        val_loader = DataLoader(valDataset_col[i], batch_size=config['training']['batch_size'], shuffle=False,
                                 num_workers = 1, persistent_workers = config['data']['dataloader']['persistent_workers'])
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
        Validate_Dataset(config,testDataset_col[i],model,loss_function)

        # Save model
        try:
            save_model(config, model, f"DlModel_Weights.pth")
            save_config(config,f"DlModel_Config.yaml")
        except:
            print("WARNING: SAVING NOT WORKING!! PLEASE CHECK")

        # Save confusion matrix
        try:
            test_loader = DataLoader(testDataset_col[i], batch_size=config['training']['batch_size'], shuffle=False,
                                 num_workers = 1, persistent_workers = config['data']['dataloader']['persistent_workers'])
            out_tot, targets_tot = get_output_results(model, test_loader, config)
            confusion_matrix_multi("confusion_matrix_TestDataset", out_tot, targets_tot, config)
        except:
            print("Error")

        # Save Datasets
        save_dataset(config,trainDataset_col[i],"train_Dataset")
        save_dataset(config,valDataset_col[i],"validation_Dataset")
        save_dataset(config,testDataset_col[i],"test_Dataset")
        save_dataset_summary(config,trainDataset_col[i],valDataset_col[i],testDataset_col[i])

        logging.info("Information about the val_auc_array:")
        logging.info(val_auc_col)
        #if(np.mean(np.array(val_auc_col)) < 0.7 and i >= 1 and len(trainDataset_col) -1 != i):
        #    logging.info(f"Stopped the folds due to a low validation AUC.")
        #    break

        # TODO: Check if run needs to be aborted
        #if(config['data']['kFolds']['skip_badtrial'] and config['run']['patienceExhausted'] and (config['run']['patienceExhaustedIndex'] < (config['training']['patience'] + 10)) and i == 0):
        #    if(len(trainDataset_col) > 1):
        #        logging.info("Patience exhausted and ignoring K-split datasets")
        #    break

    val_lossMean =  np.mean(val_loss_col)
    if(groupVar != None):
        val_lossMean =  np.mean(val_loss_col)
        logging.info(f"The mean loss of K-folds is: {round(val_lossMean,2)}")
    else:
        val_lossMean = val_loss_col[0]
        # Create a results file for the Folds results and the mean values    

    return val_lossMean




def UpdateEventTrial(hyperClass, trial, config):
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

    # Create the datasets
    datasets_col, metadata = load_dataset_total(config)
    trainDataset_col = datasets_col[0]
    valDataset_col= datasets_col[1]
    testDataset_col = datasets_col[2]

    # Check if Kfolds is active --> group
    groupVar = None
    if(len(trainDataset_col) > 1):
        groupVar = f"Folds{config['general']['trialNumber']}"


    logging.info(f"Dataset folds: {len(trainDataset_col)}")
    # Amount of Ksplits
    val_loss_col = []
    val_auc_col = []
    for i in range(len(trainDataset_col)):
        if(groupVar != None):
            CreateResultDir(config,i)

        if(trial != None and config['hyperparam_tuning']['WandB']['IsEnabled']):
            CreateStudy(config,configWandB,groupVar)

         # Get the data loaders
        train_loader = DataLoader(trainDataset_col[i], batch_size=config['training']['batch_size'], shuffle=True, 
                                      num_workers = config['data']['dataloader']['num_workers'], persistent_workers = config['data']['dataloader']['persistent_workers'])
        val_loader = DataLoader(valDataset_col[i], batch_size=config['training']['batch_size'], shuffle=False,
                                 num_workers = 1, persistent_workers = config['data']['dataloader']['persistent_workers'])
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
        val_loss, val_auc = validate_event(loss_function, model, val_loader, config)
        val_loss_col.append(val_loss)
        val_auc_col.append(val_auc)

        # Save model
        try:
            save_model(config, model, f"DlModel_Weights.pth")
            save_config(config,f"DlModel_Config.yaml")
        except:
            print("WARNING: SAVING NOT WORKING!! PLEASE CHECK")

        # Save confusion matrix
        try:
            test_loader = DataLoader(testDataset_col[i], batch_size=config['training']['batch_size'], shuffle=False,
                                 num_workers = 1, persistent_workers = config['data']['dataloader']['persistent_workers'])
            out_tot, targets_tot = get_output_results(model, test_loader, config)
            confusion_matrix_multi("confusion_matrix_TestDataset", out_tot, targets_tot, config)
        except:
            print("Error")

        # Save Datasets
        save_dataset(config,trainDataset_col[i],"train_Dataset")
        save_dataset(config,valDataset_col[i],"validation_Dataset")
        save_dataset(config,testDataset_col[i],"test_Dataset")
        save_dataset_summary(config,trainDataset_col[i],valDataset_col[i],testDataset_col[i])

        logging.info("Information about the val_auc_array:")
        logging.info(val_auc_col)
        #if(np.mean(np.array(val_auc_col)) < 0.7 and i >= 1 and len(trainDataset_col) -1 != i):
        #    logging.info(f"Stopped the folds due to a low validation AUC.")
        #    break

        # TODO: Check if run needs to be aborted
        #if(config['data']['kFolds']['skip_badtrial'] and config['run']['patienceExhausted'] and (config['run']['patienceExhaustedIndex'] < (config['training']['patience'] + 10)) and i == 0):
        #    if(len(trainDataset_col) > 1):
        #        logging.info("Patience exhausted and ignoring K-split datasets")
        #    break

    val_lossMean =  np.mean(val_loss_col)
    if(groupVar != None):
        val_lossMean =  np.mean(val_loss_col)
        logging.info(f"The mean loss of K-folds is: {round(val_lossMean,2)}")
    else:
        val_lossMean = val_loss_col[0]
        # Create a results file for the Folds results and the mean values    

    return val_lossMean






def Validate_Dataset(config, dataset, model, loss_function):
    try:
        if(dataset.df.shape[0] != 0):
            test_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False,
                                        num_workers = 1, persistent_workers = config['data']['dataloader']['persistent_workers'])
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

    # Create folder if does not exist
    create_folder(folderPath)
    check_names(folderPath)

    logging.info(f"Directory path updated: {folderPath}")
    config['general']['resultsCurrentDirectory'] = folderPath + os.sep
