# -*- coding: utf-8 -*-
import os

import sys
from pathlib import Path
path_src = os.getcwd()
sys.path.insert(1, path_src)

from src.config_presets.tools.get_config import get_config, load_config
from src.dataset.load_dataset import load_dataset, load_dataset_total
from src.models.tools.save_model import save_model
from src.training.train import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.hyper_opt.hyperHandler import HyperTuning_Handler
from src.utils.fileHandler import create_file


def main():
    
    # Setup
    # toxicity, log_level = parse_args()
    # setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = load_config('Multi_tox')

    # Disable randomness
    set_random_seed(config['general']['seed'])

    # Load the dataset
    # logging.info('Loading dataset')
    train_data, metadata = load_dataset(config, os.path.join(config['paths']['csv'], 'train_full.csv'), augment=True)
    val_data, _ = load_dataset(config, os.path.join(config['paths']['csv'], 'valid.csv'))

    # Start training
    model = train(config, train_data, val_data, metadata)

    # Save the model
    save_model(config, model, 'dl_model_full.pth')
    
    
    return



if __name__ == '__main__':
    
    main()