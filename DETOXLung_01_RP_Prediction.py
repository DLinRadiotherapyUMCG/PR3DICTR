import logging
import os

import wandb
from src.get_config import get_config
from src.load_dataset import load_dataset, load_dataset_total
from src.save_model import save_model
from src.train_multi import train
from src.utils.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.hyper_opt.hyperHandler import HyperTuning_Handler
from src.utils.fileHandler import create_file

if __name__ == '__main__':
    # Setup
    toxicity, log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = get_config('DETOXLung_config')

    # Disable randomness
    set_random_seed(config['seed'])

    # MAIN: DL running class with hyperparameter optimization
    hyperClass = HyperTuning_Handler(config)
    hyperClass.Operate(config)
    hyperClass.Stop()
