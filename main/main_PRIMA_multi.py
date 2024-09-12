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
from src.training.train_multi import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.hyper_opt.hyperHandler import HyperTuning_Handler
from src.utils.fileHandler import create_file
from src.training.train_multi import move_batch_to_device
from src.constants import DEVICE
from src.models.tools.get_multi_model import get_classification_model

from src.utils.loss_func.get_loss_function import get_loss_function

from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time


if __name__ == '__main__':
    # Setup
    toxicity, log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = get_config('Multi_tox')

    # Disable randomness
    set_random_seed(config['general']['seed'])

    # MAIN: DL running class with hyperparameter optimization
    # hyperClass = HyperTuning_Handler(config)
    # hyperClass.Operate(config)
    # hyperClass.Stop()

    # Load the dataset
    datasets_col, metadata = load_dataset_total(config)
    trainDataset_col = datasets_col[0]
    valDataset_col= datasets_col[1]
    testDataset_col = datasets_col[2]

    # Train and time a model
    for i in range(len(trainDataset_col)):

        model = get_classification_model(config, metadata, save_summary=True)  # BUG: this does not make the directories properly? (when wandb and optuna are disabled)
        model.to(device=DEVICE)

        print(model)

        loss_function = get_loss_function(config)


        ###### TEST WITH DEFAULT WORKERS (0, which means 1 thread is used) ######
        train_dataloader = DataLoader(trainDataset_col[i], batch_size=config['training']['batch_size'], shuffle=True)
        
        dataloader_times = []

        for _ in range(3):
            
            start = time()

            for batch in tqdm(train_dataloader):
                #print(batch)

                inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)
                
                outputs = model(x=inputs, features=clinical_features)

                loss = loss_function(outputs, targets) # for multi need different loss calculation
                loss.backward()
            end = time()

            print(f"Time taken (s): {end-start}")
            dataloader_times.append(end-start)

        print(f"Average time taken (s) with one dataloader: {sum(dataloader_times)/len(dataloader_times)}")
        

        ###### TEST WITH 4 PERISTENT WORKERS ######
        train_dataloader = DataLoader(trainDataset_col[i], batch_size=config['training']['batch_size'], shuffle=True, 
                                      num_workers=4, persistent_workers=True)
                                      #num_workers=config['data']['dataloader']['num_workers'], persistent_workers=config['data']['dataloader']['persistent_workers'])
        
        dataloader_times = []

        for _ in range(3):
            
            start = time()

            for batch in tqdm(train_dataloader):
                #print(batch)

                inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)
                
                outputs = model(x=inputs, features=clinical_features)

                loss = loss_function(outputs, targets) # for multi need different loss calculation
                loss.backward()
            end = time()

            print(f"Time taken (s): {end-start}")
            dataloader_times.append(end-start)

        print(f"Average time taken (s) with one dataloader: {sum(dataloader_times)/len(dataloader_times)}")



def targets_to_dict(targets, labels_list):
    targets_dict = dict()
    for i, label in enumerate(labels_list):
        targets_dict[label] = targets[:, i]
    return targets_dict