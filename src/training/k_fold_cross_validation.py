import os
import logging
import pandas as pd
import numpy as np
import torch

from src.constants import DEVICE
from src.training.train import train
from src.training.validate import validate
from src.dataset.load_dataset import load_dataset, generate_K_fold_cross_validation_splits
from src.models.tools.get_classification_model import get_classification_model
from src.dataset.get_dataloader import make_dataloader   
from src.dataset.get_transforms import get_transforms
from src.utils.loss_func.get_loss_function import get_loss_function
from src.training.tools.saving_predictions import concatenate_predictions, save_predictions
from src.hyper_opt.hyperHandler import CreateResultDir



def K_fold_cross_validation(config, hyperClass=None):

    endpoint_list = config['columns']['label']

    # load the data
    df_train_val, df_test = load_dataset(config)
    if config['general']['use_test_set']:
        test_loader, _ = make_dataloader(config, df_test, val_transforms, validation_mode=True)

    # make K fold splits
    k_fold_dataframes_list = generate_K_fold_cross_validation_splits(config, df_train_val)

    # get the data transforms
    train_transforms, val_transforms = get_transforms(config)
    # get the loss function
    loss_function = get_loss_function(config)

    # iterate through the folds 
    for fold_idx, dataset_split_dict in enumerate(k_fold_dataframes_list):
        CreateResultDir(config, fold_idx)


        train_data, val_data = dataset_split_dict['train'], dataset_split_dict['val']

        # make the dataloaders for this fold
        train_loader, metadata = make_dataloader(config, train_data, train_transforms, validation_mode=False)
        val_loader, _ = make_dataloader(config, val_data, val_transforms, validation_mode=True)

        # initialise a model
        logging.info('Getting model')
        model = get_classification_model(config, metadata=metadata, save_summary=True)
        model.to(device=DEVICE)

        # train the model
        model = train(config, model, loss_function, train_loader, val_loader, hyperClass=hyperClass)
        model.eval()

        # get the predictions of the trained model on the training and validation (and test) sets
        logging.info('Getting predictions')
        logging.info('   training set')
        train_loss, train_auc_dict, train_preds_dict, train_targets_dict, train_patientIDs_list = validate(config, model, loss_function, train_loader)
        logging.info('   validation set')
        val_loss, val_auc_dict, val_preds_dict, val_targets_dict, val_patientIDs_list = validate(config, model, loss_function, val_loader)

        if config['general']['use_test_set']:
            logging.info('   test set')
            test_loss, test_auc_dict, test_preds_dict, test_targets_dict, test_patientIDs_list = validate(config, model, loss_function, test_loader)
        else:
            test_loss = None
            test_auc_dict = None
            test_patientIDs_list = []
            test_preds_dict = {endpoint: [] for endpoint in endpoint_list}
            test_targets_dict = {endpoint: [] for endpoint in endpoint_list}

        # concatenate all of the predictions
        print("Concatenating predictions")
        all_preds_dict, all_targets_dict = concatenate_predictions(config, [train_preds_dict, val_preds_dict, test_preds_dict], [train_targets_dict, val_targets_dict, test_targets_dict])
        #all_targets_dict = concatenate_predictions(config, )
        all_patientIDs_list = train_patientIDs_list + val_patientIDs_list + test_patientIDs_list
        mode_list = ['train']*len(train_patientIDs_list) + ['val']*len(val_patientIDs_list) + ['test']*len(test_patientIDs_list)

        # save the predictions
        print("Saving predictions")
        save_predictions(config, all_patientIDs_list, all_preds_dict, all_targets_dict, mode_list, fold_idx)


        # save the results # NOTE: how do we want to do this?
        # hyperhandler-like stuff here?




def save_model_predictions():
    pass