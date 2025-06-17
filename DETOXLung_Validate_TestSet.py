import logging
import os

import wandb

# Set working directory to --> "Pred_RT"
import sys
from pathlib import Path
path_src = os.getcwd()
sys.path.insert(1, path_src)

from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset, load_dataset_total
from src.models.tools.save_model import save_model
from src.training.train import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.hyper_opt.hyperHandler import HyperTuning_Handler
from src.utils.fileHandler import create_file, create_textfile

from src.evaluation.validate_on_test_set import validate_models_on_test_set

if __name__ == '__main__':
    # Setup
    log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()v
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = get_config('DETOXLung_config')

    # Disable randomness
    set_random_seed(config['general']['seed'])


    trial_dir = r"/scratch/p315317/Project/DETOX_Lung_RP/Results/Habrok_ResNet18-02_2025/Trial_33" # config['general']['resultsCurrentDirectory']
    # # run the models on the test set
    validate_models_on_test_set(config, trial_dir)
    # MAIN: DL running class with hyperparameter optimization
    #hyperClass = HyperTuning_Handler(config)
    #hyperClass.Operate(config)
    #hyperClass.Stop()
