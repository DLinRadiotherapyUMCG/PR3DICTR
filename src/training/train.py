import os
import logging
import pandas as pd
import numpy as np
import time
import torch
import wandb
import copy
from tqdm import tqdm


from src.constants import DEVICE
from src.utils.loss_func.get_loss_function import get_loss_function
from src.utils.optimizer.get_optimizer import get_optimizer
from src.utils.scheduler.get_scheduler import get_scheduler
from src.evaluation.calculate_auc import calculate_auc_multi
from src.training.tools.utils import move_batch_to_device
from src.training.validate import validate
from src.models.tools.save_model import save_model, load_model


from src.hyper_opt.WandB_hpt import WandB_is_enabled
from src.visualization.plot_slices import plot_slices

import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def plot_model_inputs(config, plot_inputs, epoch):
    """
    Plots the model inputs (CT, RTDOSE, Segmentation) used during training, and saves the figure as a png.
    Args:
        config (ConfigObject): configuration object
        plot_inputs (list): list of model input images
        epoch (int): current epoch number
    """
    for patient_idx in range(min(1, len(plot_inputs))):
        
        ct_range = (config['data']['preprocessing']['ct']['a_max'] - config['data']['preprocessing']['ct']['a_min'])
        rtdose_range = (config['data']['preprocessing']['rtdose']['a_max'] - config['data']['preprocessing']['rtdose']['a_min'])
        
        CT = (plot_inputs[patient_idx][0].cpu() * ct_range) + config['data']['preprocessing']['ct']['a_min']
        RTDOSE = plot_inputs[patient_idx][1].cpu() * rtdose_range
        RTSTRUCT = plot_inputs[patient_idx][2].cpu()

        plotting_rows_dicts = [
            {
                "Label": "CT",
                "CT": CT
            },
            {
                "Label": "RTDOSE",
                "RTDOSE": RTDOSE
            },
            {
                "Label": "RTSTRUCT",
                "RTSTRUCT": RTSTRUCT
            }
        ]

        slices = [20, 30, 40, 50, 60, 70, 80, 90]
        fig, axes = plot_slices(plotting_rows_dicts, slices)# , title=f"{patient_id} slices")

        filename=os.path.join(config['general']['resultsCurrentDirectory'],'epoch_{}_idx_{}.png'.format(epoch, patient_idx))
        #print(filename)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)



def train(config, model, loss_function, train_loader, val_loader, hyperClass = None):
    """
    Train the model.
    :param config:
    :param train_data:
    :param val_data:
    :return: Model
    """

    # config run information
    config['run']['patienceExhausted'] = False
    config['run']['patienceExhaustedIndex'] = 0
    show_pbar = config['training']['progress_bar']
    
    # Get the names of the end-points being evaluated 
    labels = config['columns']['label']

    # Get loss function, optimizer, and scheduler
    loss_function = get_loss_function(config)
    optimizer = get_optimizer(config, model)
    if config['training']['scheduler']['name'] is not False:
        scheduler = get_scheduler(config, optimizer)

    # Initialize the best model and lowest validation loss
    best_model_state_dict = None
    if(config['general']['optimize'] == "AUC"):
        print("Optimizing based on AUC")
        lowest = 0
    else:
        lowest = np.inf
    patience_counter = 0
    
    #Sigmoid = np.vectorize(sigmoid)

    sigmoid_act = torch.nn.Sigmoid()

    # Training loop
    logging.info('Starting training loop')
    for epoch_num in range(1, config['training']['max_epochs'] + 1):
        #current_epoch_num = epoch+1
        resultsEpoch = dict()
        out_tot = dict.fromkeys(labels)
        targets_tot = dict.fromkeys(labels)
        
        for label in labels:
            out_tot[label] = []
            targets_tot[label] = []

        logging.info(f'Epoch {epoch_num}')
        model.train()

        total_loss = 0.0
        num_batches_per_epoch = len(train_loader)

        start_epoch_time = time.time()
  
        #it = tqdm(len(train_loader))
        if show_pbar:
            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch_num}', position=0, leave=True)

        for batch_num, batch in enumerate(train_loader, start=1):  # , disable=config["hyperparam_tuning"]["optuna"]["IsEnabled"]
            logging.debug(f'Batch {batch_num} of epoch {epoch_num}')

            if show_pbar: pbar.update(1)
            #it.update(1)
            
            # if batch_num == 0:
            #     print("IDs first batch: ", batch['patient_id'])

            optimizer.zero_grad(set_to_none=True)
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

            # Make predictions
            outputs = model(x=inputs, features=clinical_features)

            # Calculate loss
            loss = loss_function(outputs, targets) # for multi need different loss calculation
            if loss.grad_fn:
                loss.backward()

            # Calculate AUC            
            for idx, label in enumerate(labels):
                out_tot[label] = out_tot[label] + list(sigmoid_act(outputs[label]).cpu().detach().numpy().reshape((1,targets[:,idx].shape[0]))[0])
                targets_tot[label] = targets_tot[label] + list(targets[:,idx].cpu().detach().numpy().reshape((1,targets[:,idx].shape[0]))[0])

            # Log loss and AUC
            total_loss += loss.item()
            # if auc is not None:
            #     total_auc += auc
            #     num_auc_batches += 1

            # Update model weights
            optimizer.step()

            # Step the scheduler
            if config['training']['scheduler']['name'] in ['cosine', 'exponential']:  # , 'step']:
                scheduler.step(epoch_num + (batch_num / num_batches_per_epoch))
            elif config['training']['scheduler']['name'] in ['cyclic']:
                scheduler.step()


            # plot model inputs
            #print(epoch_num)
            #print(batch_num)
            if epoch_num == 1 and batch_num==1:
                plot_model_inputs(config=config, plot_inputs=inputs, epoch=epoch_num)     

        if show_pbar: pbar.close()

        #it.close()
        auc = calculate_auc_multi(out_tot, targets_tot, config)
            
        #logging.info('Training AUCs')
        #logging.info(auc)
        # Log epoch loss and AUC
        avg_loss = total_loss / num_batches_per_epoch
        logging.info(f'  Training   Loss={avg_loss:.5f}, AUCs={auc}')

        resultsEpoch.update({'Train_Loss':avg_loss})
        keys = list(auc.keys())
        values = list(auc.values())
        for i in range(len(keys)):
            resultsEpoch.update({"Train_AUC_"+keys[i]:values[i]})
        resultsEpoch.update({"Train_AUC" : np.mean(values)})

        

        # Perform validation
        if epoch_num % config['training']['validation_interval'] == 0:
            val_loss, auc_val, val_preds_dict, val_labels_dict, val_patientIDs_list = validate(config, model, loss_function, val_loader)
            # wandb.log({'train/loss': avg_loss, 'train/auc': avg_auc, 'val/loss': val_loss, 'val/auc': val_auc})
            logging.info(f'  Validation Loss={val_loss:.5f}, AUCs={auc_val}')
            resultsEpoch.update({'Validation_Loss':val_loss})

            keys = list(auc_val.keys())
            values = list(auc_val.values())
            for i in range(len(keys)):
                resultsEpoch.update({"Validation_AUC_"+keys[i] : values[i]})
            resultsEpoch.update({"Validation_AUC" : np.mean(values)})

            # Check if this model has the lowest validation loss        # TODO: turn this into a function? it's basically the same code twice
            if(config['general']['optimize'] == "AUC"):
                if val_loss[0] > lowest:
                    logging.info(f'New highest AUC: {values[0]}')
                    lowest = values[0]
                    patience_counter = 0 # Reset patience counter

                    #best_model_state_dict = copy.deepcopy(model.state_dict())
                    if config['Save']['best_model']: 
                        save_model(config, model, f"DlModel_Weights.pth")
                    else:
                        best_model_state_dict = copy.deepcopy(model.state_dict())  # must do a deepcopy of the state_dict to avoid the model being updated
                    
                else:
                    patience_counter += 1 # Increment patience counter
            else:
                # Check if this model has the lowest validation loss
                if val_loss < lowest:
                    logging.info(f'New lowest loss: {val_loss}')
                    lowest = val_loss
                    patience_counter = 0 # Reset patience counter
                    
                    if config['Save']['best_model']: 
                        save_model(config, model, f"DlModel_Weights.pth")
                    else:
                        best_model_state_dict = copy.deepcopy(model.state_dict())
                    
                else:
                    patience_counter += 1 # Increment patience counter

            # Check if patience has been exhausted
            if patience_counter >= config['training']['patience']:
                logging.info('Patience exhausted, stopping training')
                config['run']['patienceExhausted'] = True
                config['run']['patienceExhaustedIndex'] = epoch_num
                break
        
        logging.info(f'  Epoch duration = {(time.time() - start_epoch_time):.2f} seconds')


        if WandB_is_enabled(config):
            resultsEpoch.update({"epoch":epoch_num})
            wandb.log(resultsEpoch)
        
        # else:
        #     wandb.log({'train/loss': avg_loss, 'train/auc': avg_auc})

    # Log highest AUC
    # wandb.log({'train/highest_auc': highest_auc})
    # logging.info('Finished training')

    #model = load_model(config, model, f"DlModel_Weights.pth")
    if config['Save']['best_model']: 
        model = load_model(config, model, f"DlModel_Weights.pth")
    else:
        model.load_state_dict(best_model_state_dict)    

    return model







def get_output_results(model, val_loader, config):
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
