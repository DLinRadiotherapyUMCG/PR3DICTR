import logging
import os

import wandb

# Set working directory to --> "Pred_RT"
import sys
from pathlib import Path
path_src = os.getcwd()
sys.path.insert(1, path_src)

from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset, check_image_data_exists
from src.models.tools.save_model import save_model
from src.training.train import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from pred_RT.src.hyper_opt.OptunaExperimentManager import OptunaExperimentManager
from src.utils.fileHandler import create_file, create_textfile
from src.experiments.experimentHandler import experimentHandler 

from sklearn.metrics import accuracy_score, roc_auc_score

if __name__ == '__main__':
    # Setup
    log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = get_config('DETOXLung_config')

    # Disable randomness
    set_random_seed(config['general']['seed'])

    # MAIN: DL running class with hyperparameter optimization
    hyperClass = experimentHandler(config)
    hyperClass.run_experiment(config)
    #hyperClass.Operate(config)
    #hyperClass.Stop()


    # There are 3 options to get the train, validation data
    #   1) There are 2 seperate files
    #   2) Single file that contains splitVar
    #   3) Single file with no splitVar will custom be split in --> train,var,test (warning: test need to be unique by seed --> need to check)

    # Check if 1 single file is given (testData is not always available)
