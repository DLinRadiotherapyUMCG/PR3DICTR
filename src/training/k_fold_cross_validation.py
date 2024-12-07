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
from src.utils.saving.create_results_directory import create_results_directory
from src.utils.list_dicts import append_to_list_dicts


from src.hyper_opt.WandB_hpt import initialise_WandB_group, login, WandB_stop
from src.evaluation.mainMetricHandler import mainMetricHandler


def K_fold_cross_validation(config, config_for_wandb=None):
    """
    desc.
    """
    n_splits = config['data']['kFolds']['n_splits']
    n_iterations = config['data']['kFolds']['n_iterations']
    if n_splits < n_iterations:
        raise ValueError("The number of k-fold splits must be greater than or equal to the number of k-fold iterations.")
    elif n_iterations == 0:
        raise ValueError("The number of k-fold iterations must be greater than 0.")

    endpoint_list = config['columns']['labels']

    train_aucs_list = []
    val_aucs_list = []
    test_aucs_list = []
    train_aucs_list_dict = {endpoint: list() for endpoint in endpoint_list}
    val_aucs_list_dict   = {endpoint: list() for endpoint in endpoint_list}
    test_aucs_list_dict  = {endpoint: list() for endpoint in endpoint_list}

    train_losses_list = []
    val_losses_list = []
    test_losses_list = []
    train_losses_list_dict = {endpoint: list() for endpoint in endpoint_list}
    val_losses_list_dict   = {endpoint: list() for endpoint in endpoint_list}
    test_losses_list_dict  = {endpoint: list() for endpoint in endpoint_list}
    


    # load the data
    df_train_val, df_test = load_dataset(config)
    if config['general']['use_test_set']:
        test_loader, _ = make_dataloader(config, df_test, val_transforms, validation_mode=True)

    # make K fold splits
    k_fold_dataframes_list = generate_K_fold_cross_validation_splits(config, df_train_val)
    
    # cap the number of iterations, if it is less than the number of k-splits to make
    n_iterations = config['data']['kFolds']['n_iterations']
    if n_iterations < len(k_fold_dataframes_list):
        k_fold_dataframes_list = k_fold_dataframes_list[:n_iterations]

    # get the data transforms
    train_transforms, val_transforms = get_transforms(config)
    # get the loss function
    loss_function = get_loss_function(config)


    # iterate through the folds 
    for fold_idx, dataset_split_dict in enumerate(k_fold_dataframes_list, start=1):
        
        logging.info(f'Fold {fold_idx}/{len(k_fold_dataframes_list)}')
        
        initialise_WandB_group(config, project_name=config['general']['experiment_name'], groupName=config['general']['trialNumber'], config_for_wandb=config_for_wandb)

        create_results_directory(config, fold_idx)

        train_data, val_data = dataset_split_dict['train'], dataset_split_dict['val']

        # make the dataloaders for this fold
        train_loader, metadata = make_dataloader(config, train_data, train_transforms, validation_mode=False)
        val_loader, _ = make_dataloader(config, val_data, val_transforms, validation_mode=True)

        metricHandler = mainMetricHandler(config)


        # initialise a model
        logging.info('Getting model')
        model = get_classification_model(config, metadata=metadata, save_summary=True)
        model.to(device=DEVICE)

        # train the model
        model = train(config, model, loss_function, train_loader, val_loader, metricHandler)
        model.eval()

        # get the predictions of the trained model on the training and validation (and test) sets
        logging.info('Getting predictions of best model from this fold:')
        logging.info('   Training set')
        train_loss, train_mean_metric_val, train_metric_dict, train_preds_dict, train_targets_dict, train_patientIDs_list = validate(config, model, loss_function, train_loader, metricHandler)
        print("   ", train_loss, train_mean_metric_val, train_metric_dict)
        logging.info('   Validation set')
        val_loss, val_mean_metric_val, val_metric_dict, val_preds_dict, val_targets_dict, val_patientIDs_list = validate(config, model, loss_function, val_loader, metricHandler)
        print("   ",val_loss, val_mean_metric_val, val_metric_dict)

        if config['general']['use_test_set']:
            logging.info('   Test set')
            test_loss, test_mean_metric_val, test_metric_dict, test_preds_dict, test_targets_dict, test_patientIDs_list = validate(config, model, loss_function, test_loader, metricHandler)
            print("   ",test_loss, test_mean_metric_val, test_metric_dict)
        else:
            test_loss = None
            test_mean_metric_val = None
            test_metric_dict = {endpoint: None for endpoint in endpoint_list}
            test_patientIDs_list = []
            test_preds_dict = {endpoint: [] for endpoint in endpoint_list}
            test_targets_dict = {endpoint: [] for endpoint in endpoint_list}
        

        """    
        Handling Results  
        """

        # concatenate all of the predictions
        all_preds_dict, all_targets_dict = concatenate_predictions(config, [train_preds_dict, val_preds_dict, test_preds_dict], [train_targets_dict, val_targets_dict, test_targets_dict])
        #all_targets_dict = concatenate_predictions(config, )
        all_patientIDs_list = train_patientIDs_list + val_patientIDs_list + test_patientIDs_list
        mode_list = ['train']*len(train_patientIDs_list) + ['val']*len(val_patientIDs_list) + ['test']*len(test_patientIDs_list)

        # save the predictions
        save_predictions(config, all_patientIDs_list, all_preds_dict, all_targets_dict, mode_list, fold_idx)


        # concatenate all of the AUC dicts and loss dicts
        [train_aucs_list_dict, val_aucs_list_dict, test_aucs_list_dict] = append_to_list_dicts(config, 
                                                                                               [train_aucs_list_dict, val_aucs_list_dict, test_aucs_list_dict], 
                                                                                               [train_metric_dict, val_metric_dict, test_metric_dict]
                                                                                               )
       
        # [train_losses_list_dict, val_losses_list_dict, test_losses_list_dict] = append_to_list_dicts(config,
        #                                                                                                 [train_losses_list_dict, val_losses_list_dict, test_losses_list_dict],
        #                                                                                                 [train_loss, val_loss, test_loss]
        #                                                                                                 )
        
        train_losses_list.append(train_loss)
        val_losses_list.append(val_loss)
        test_losses_list.append(test_loss)
        
        
        # save the results # NOTE: how do we want to do this?
        # hyperhandler-like stuff here?

        # stop WandB for this fold (init a new one on the next fold)
        WandB_stop(config)


    
    # compute mean AUC per endpoint
    train_aucs_mean_dict = {endpoint: np.mean(aucs) for endpoint, aucs in train_aucs_list_dict.items()}
    val_aucs_mean_dict = {endpoint: np.mean(aucs) for endpoint, aucs in val_aucs_list_dict.items()}
    # compute total mean AUC
    mean_train_auc = np.mean(list(train_aucs_mean_dict.values()))
    mean_val_auc = np.mean(list(val_aucs_mean_dict.values()))

    # compute mean loss per endpoint
    train_losses_mean_dict = {endpoint: np.mean(losses) for endpoint, losses in train_losses_list_dict.items()}
    val_losses_mean_dict = {endpoint: np.mean(losses) for endpoint, losses in val_losses_list_dict.items()}
    # compute total mean loss
    mean_train_loss = np.mean(train_losses_list)
    mean_val_loss = np.mean(val_losses_list)

    # do that for the test set
    if config['general']['use_test_set']:
        test_aucs_mean_dict = {endpoint: np.mean(aucs) for endpoint, aucs in test_aucs_list_dict.items()}
        mean_test_auc = np.mean(list(test_aucs_mean_dict.values()))
        test_losses_mean_dict = {endpoint: np.mean(losses) for endpoint, losses in test_losses_list_dict.items()}
        mean_test_loss = np.mean(test_losses_list)
    else:
        test_aucs_mean_dict = {endpoint: None for endpoint in endpoint_list}
        mean_test_auc = None
        test_losses_mean_dict = {endpoint: None for endpoint in endpoint_list}
        mean_test_loss = None




    results = { # mean AUC
                "train_mean_AUC": mean_train_auc,
                "val_mean_AUC": mean_val_auc,
                "test_mean_AUC": mean_test_auc,
                # mean Loss
                "train_mean_loss": mean_train_loss,
                "val_mean_loss": mean_val_loss,
                "test_mean_loss": mean_test_loss,
               }
    
    for endpoint in endpoint_list:
        # mean AUC per endpoint
        results[f"train_{endpoint}_AUC"] = train_aucs_mean_dict[endpoint]
        results[f"val_{endpoint}_AUC"] = val_aucs_mean_dict[endpoint]
        results[f"test_{endpoint}_AUC"] = test_aucs_mean_dict[endpoint]

        # mean loss per endpoint
        results[f"train_{endpoint}_loss"] = train_losses_mean_dict[endpoint]
        results[f"val_{endpoint}_loss"] = val_losses_mean_dict[endpoint]
        results[f"test_{endpoint}_loss"] = test_losses_mean_dict[endpoint]


    return results

    

