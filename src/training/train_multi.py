import os
import logging

import torch
import wandb
from torch.utils.data import DataLoader

from src.constants import DEVICE
from src.utils.loss_func.get_loss_function import get_loss_function
from src.models.tools.get_multi_model import get_classification_model
from src.utils.optimizer.get_optimizer import get_optimizer
from src.utils.scheduler.get_scheduler import get_scheduler
from src.evaluation.calculate_auc import calculate_auc
from src.evaluation.calculate_auc import calculate_auc_multi

import pandas as pd
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def train(config, train_loader, val_loader, metadata, hyperClass = None):
    """
    Train the model.
    :param config:
    :param train_data:
    :param val_data:
    :param metadata:
    :return: Model
    """

    # config run information
    config['run']['patienceExhausted'] = False
    config['run']['patienceExhaustedIndex'] = 0
    
    # Get the names of the end-points being evaluated 
    labels = config['columns']['label']

    # Get the model
    logging.info('Getting model')
    model = get_classification_model(config, metadata)
    model.to(device=DEVICE)
    # wandb.watch(model, log_freq=100)

    # Get loss function, optimizer, and scheduler
    loss_function = get_loss_function(config)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    # Initialize the best model and lowest validation loss
    best_model = None
    lowest_loss = np.inf
    patience_counter = 0
    
    Sigmoid = np.vectorize(sigmoid)

    # Training loop
    logging.info('Starting training loop')
    for epoch in range(config['training']['max_epochs']):
        resultsEpoch = dict()
        out_tot = dict.fromkeys(labels)
        targets_tot = dict.fromkeys(labels)
        
        for label in labels:
            out_tot[label] = []
            targets_tot[label] = []

        logging.info(f'Starting epoch {epoch}')
        model.train()

        total_loss = 0.0
        total_auc = 0.0
        num_batches = 0
        num_auc_batches = 1

        for i, batch in enumerate(train_loader):
            logging.debug(f'Batch {i} of epoch {epoch}')

            optimizer.zero_grad(set_to_none=True)
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

            # Make predictions
            outputs = model(x=inputs, features=clinical_features)

            # Calculate loss
            loss = loss_function(outputs, targets) # for multi need different loss calculation
            loss.backward()

            # Calculate AUC            
            for idx, label in enumerate(labels):
                out_tot[label] = out_tot[label] + list(Sigmoid(outputs[label].cpu().detach().numpy().reshape((1,targets[:,:,idx].shape[0]))[0]))
                targets_tot[label] = targets_tot[label] + list(targets[:,:,idx].cpu().detach().numpy().reshape((1,targets[:,:,idx].shape[0]))[0])

            # Log loss and AUC
            total_loss += loss.item()
            # if auc is not None:
            #     total_auc += auc
            #     num_auc_batches += 1

            # Update model weights
            optimizer.step()

            # Step the scheduler
            scheduler.step(epoch + (i + 1) / len(train_loader))

            num_batches += 1      

        auc = calculate_auc_multi(out_tot,targets_tot,config)
            
        logging.info('Training AUCs')
        logging.info(auc)
        # Log epoch loss and AUC
        avg_loss = total_loss / num_batches
        logging.info(f'Epoch loss: {avg_loss}')

        resultsEpoch.update({'Train_Loss':avg_loss})
        keys = list(auc.keys())
        values = list(auc.values())
        for i in range(len(keys)):
            resultsEpoch.update({"Train_AUC_"+keys[i]:values[i]})

        # Perform validation
        if epoch % config['training']['validation_interval'] == 0:
            val_loss, auc_val = validate(loss_function, model, val_loader,config)
            # wandb.log({'train/loss': avg_loss, 'train/auc': avg_auc, 'val/loss': val_loss, 'val/auc': val_auc})
            logging.info(val_loss)
            logging.info(auc_val)
            resultsEpoch.update({'Validation_Loss':val_loss})

            keys = list(auc_val.keys())
            values = list(auc_val.values())
            for i in range(len(keys)):
                resultsEpoch.update({"Validation_AUC_"+keys[i]:values[i]})

            # Check if this model has the lowest validation loss
            if val_loss < lowest_loss:
                logging.info(f'New lowest loss: {val_loss}')
                lowest_loss = val_loss
                best_model = model.state_dict()  # Save the model state
                patience_counter = 0 # Reset patience counter
            else:
                patience_counter += 1 # Increment patience counter

            # Check if patience has been exhausted
            if patience_counter >= config['training']['patience']:
                logging.info('Patience exhausted, stopping training')
                config['run']['patienceExhausted'] = True
                config['run']['patienceExhaustedIndex'] = epoch
                break
        
        pd.DataFrame(out_tot).to_csv(os.path.join(config['general']['resultsCurrentDirectory'],'temp_pred.csv'), sep = ';')
        pd.DataFrame(targets_tot).to_csv(os.path.join(config['general']['resultsCurrentDirectory'],'temp_target.csv'), sep = ';')
        
        # Log to hyperparam class if necessary
        if(hyperClass != None):
            resultsEpoch.update({"epoch":epoch})
            hyperClass.UpdateWandB(resultsEpoch,epoch)
        
        # else:
        #     wandb.log({'train/loss': avg_loss, 'train/auc': avg_auc})

    # Log highest AUC
    # wandb.log({'train/highest_auc': highest_auc})
    # logging.info('Finished training')



    model.load_state_dict(best_model)

    return model


def validate(loss_function, model, val_loader, config):
    model.eval()
    total_loss = 0.0
    total_auc = 0.0
    num_batches = 0
    labels = config['columns']['label']
    
    out_tot = dict.fromkeys(labels)
    targets_tot = dict.fromkeys(labels)
    
    for label in labels:
        out_tot[label] = []
        targets_tot[label] = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            logging.debug(f'Validation batch {i}')
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

            outputs = model(x=inputs, features=clinical_features)
            loss = loss_function(outputs, targets)

            total_loss += loss.item()
            
            for lab_indx, label in enumerate(labels):
                out_tot[label] = out_tot[label] + list(outputs[label].cpu().detach().numpy().reshape((1,targets[:,:,lab_indx].shape[0]))[0])
                targets_tot[label] = targets_tot[label] + list(targets[:,:,lab_indx].cpu().detach().numpy().reshape((1,targets[:,:,lab_indx].shape[0]))[0])
            
            num_batches += 1

    auc = calculate_auc_multi(out_tot,targets_tot,config)
    model.train()

    avg_loss = total_loss / num_batches
    

    logging.debug(f'Validation loss: {avg_loss}')
    # logging.debug(f'Validation AUC: {avg_auc}')

    return avg_loss, auc

def move_batch_to_device(batch, device):
    inputs, clinical_features, targets = batch
    inputs = inputs.to(device=device)
    clinical_features = clinical_features.to(device=device)
    targets = targets.to(device=device)
    return inputs, clinical_features, targets

def get_output_results(model, val_loader, config): # this function is redundant, could just be an option for the validate function
    model.eval()
    labels = config['columns']['label']
      
    out_tot = dict.fromkeys(labels)
    targets_tot = dict.fromkeys(labels)

    for label in labels:
        out_tot[label] = []
        targets_tot[label] = []
    
    Sigmoid = np.vectorize(sigmoid)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            logging.debug(f'Validation batch {i}')
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

            outputs = model(x=inputs, features=clinical_features)

            lab_indx = 0
            for label in labels:
                out_tot[label] = out_tot[label] + list(Sigmoid(outputs[label].cpu().detach().numpy().reshape((1,targets[:,:,lab_indx].shape[0]))[0]))
                targets_tot[label] = targets_tot[label] + list(targets[:,:,lab_indx].cpu().detach().numpy().reshape((1,targets[:,:,lab_indx].shape[0]))[0])
                lab_indx+=1
    return(out_tot, targets_tot)
