import logging
import os

import sys
from pathlib import Path
path_Project = Path(os.getcwd())
path_Git = path_Project.parent

def IncludeDirectory(path, index = 0, indexStop = -1):
    if(os.path.exists(path) and (index <= indexStop or indexStop == -1)):
        print(path)
        sys.path.append(path)
        children = os.listdir(path)
        for i in range(len(children)):
            pathChild = os.path.join(path,children[i])
            if(os.path.isdir(pathChild)):
                IncludeDirectory(pathChild, index + 1, indexStop)

# Activate the function
IncludeDirectory(path_Git,0,1)

import wandb
from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset, load_dataset_total
from src.models.tools.save_model import save_model
from src.training.train import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from pred_RT.src.hyper_opt.OptunaExperimentManager import OptunaExperimentManager
from src.utils.fileHandler import create_file

import sqlite3
import matplotlib.pyplot as plt
import optuna
import pandas as pd

if __name__ == '__main__':
    # Setup
    toxicity, log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    configName = 'DETOXLung_config'
    config = get_config('DETOXLung_config')

    # Disable randomness
    set_random_seed(config['general']['seed'])

    # Test a chosen project
    nameProjectTest = "DETOX-Lung_Project_7"
    config['general']['experiment_name'] = nameProjectTest
    #config['hyperparam_tuning']['optuna']['studyname'] = nameProjectTest

    #Load the existing project
    hyperClass = OptunaExperimentManager(config)
    df = hyperClass.Optuna_study.trials_dataframe()

    # Show the project information
    print(df)

    # Close the project envrionment
    hyperClass.Stop()