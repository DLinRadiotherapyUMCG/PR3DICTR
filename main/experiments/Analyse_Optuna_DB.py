import logging
import os

import wandb
from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset, load_dataset_total
from src.models.tools.save_model import save_model
from src.training.train_multi import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.hyper_opt.hyperHandler import HyperTuning_Handler
from src.utils.fileHandler import create_file

import matplotlib.pyplot as plt


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
    set_random_seed(config['seed'])

    # Test a chosen project
    nameProjectTest = "DETOX-Lung_Project_7"
    config['hyperparam_tuning']['ProjectName'] = nameProjectTest
    config['hyperparam_tuning']['optuna']['studyname'] = nameProjectTest

    #Load the existing project
    hyperClass = HyperTuning_Handler(config)
    df = hyperClass.Optuna_study.trials_dataframe()

    # Show the project information
    print(df)

    # Close the project envrionment
    hyperClass.Stop()