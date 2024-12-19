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
from src.utils.move_batch_to_device import move_batch_to_device
from src.training.validate import validate
from src.models.tools.save_model import save_model, load_model
from src.training.utils.check_improvement import check_improvement

from src.hyper_opt.WandB_hpt import WandB_is_enabled, update_WandB_summary_table
from src.visualization.plot_model_inputs import plot_model_inputs
from src.evaluation.mainMetricHandler import mainMetricHandler
from src.utils.loss_func.calc_mixup_loss import calc_mixup_loss
from src.constants import MISSING_DATA_VALUE

def train(config, model, loss_function, train_loader, val_loader, metricHandler):
    """
    Train the model.
    :param config:
    :param train_data:
    :param val_data:
    :return: Model
    """

    # config run information
    show_pbar = config['training']['show_progress_bar']

    mixup_is_enabled = config['data']['augmentation']['mixup']['isEnabled']

    # Get the names of the end-points being evaluated 
    labels = config['columns']['labels']

    # Get loss function, optimizer, and scheduler
    loss_function = get_loss_function(config)
    optimizer = get_optimizer(config, model)
    if config['training']['scheduler']['name'] is not False:
        scheduler = get_scheduler(config, optimizer)

    # get the main metric with which to evaluate the training loop (e.g. AUC)
    #metricHandler = mainMetricHandler(config)
    metric_name = metricHandler.metric_name

    # Initialize the best model and lowest validation loss
    best_model_state_dict = None
    if(config['training']['stopping_criteria'] == "loss"):
        print("Optimizing based on loss")
        best_value = np.inf
    else:
        best_value = 0
        
    patience_counter = 0
    

    sigmoid_act = torch.nn.Sigmoid()

    # Training loop
    logging.info('Starting training loop')
    for epoch_num in range(1, config['training']['max_epochs'] + 1):
        mixup_lambda_epoch = None
        #current_epoch_num = epoch+1
        improved = False # Flag to indicate if the model has improved on this epoch
        results_log = dict()
        best_log_dict = None
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
  
        if show_pbar:
            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch_num}', position=0, leave=True)

        for batch_num, batch in enumerate(train_loader, start=1):  # , disable=config["hyperparam_tuning"]["optuna"]["IsEnabled"]
            logging.debug(f'Batch {batch_num} of epoch {epoch_num}')

            if show_pbar: pbar.update(1)

            optimizer.zero_grad(set_to_none=True)
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)
            if mixup_is_enabled:
                mixup_lambda = batch['lambda']
                mixup_indices_batch = batch['indices']
                
                cur_batch_size = inputs.shape[0]
                mixup_lambda_batch = [mixup_lambda] * cur_batch_size
                mixup_adjusted_indices_batch = (batch_num-1) * cur_batch_size + mixup_indices_batch
                #mixup_indices_batch = mixup_indices_batch.to(DEVICE)

                if mixup_lambda_epoch == None:
                    mixup_lambda_epoch = torch.tensor(mixup_lambda_batch, device=DEVICE)
                    mixup_indices_epoch = mixup_adjusted_indices_batch
                else:
                    mixup_lambda_epoch = torch.cat([mixup_lambda_epoch, torch.tensor(mixup_lambda_batch, device=DEVICE)], dim=0)
                    mixup_indices_epoch = torch.cat([mixup_indices_epoch, mixup_adjusted_indices_batch], dim=0)

                # mask out missing values
                targets[targets == MISSING_DATA_VALUE] = 0

            #print("hi")
            # plot model inputs
            if (config['saving']['plot_training_slices']['isEnabled']) and (epoch_num == 1 or epoch_num % config['saving']['plot_training_slices']['every_n_epochs'] == 0) and (batch_num == 1):
                plot_model_inputs(config=config, plot_inputs=inputs, epoch=epoch_num)

            # Make predictions
            outputs = model(x=inputs, features=clinical_features)

            # Calculate loss
            if mixup_is_enabled:
                # the loss formula is different for mixup
                loss = calc_mixup_loss(outputs, targets, loss_function, mixup_indices_batch, mixup_lambda)                
            else:
                # normal loss calculation
                loss = loss_function(outputs, targets) 

            if loss.grad_fn:
                loss.backward()

            # Calculate AUC            
            for idx, label in enumerate(labels):
                out_tot[label] = out_tot[label] + list(sigmoid_act(outputs[label]).cpu().detach().numpy().reshape((1,targets[:,idx].shape[0]))[0])
                targets_tot[label] = targets_tot[label] + list(targets[:,idx].cpu().detach().numpy().reshape((1,targets[:,idx].shape[0]))[0])

            # Log the batch loss to the epoch loss
            total_loss += loss.item()

            # Update model weights
            optimizer.step()

            # Step the scheduler
            if config['training']['scheduler']['name'] in ['cosine', 'exponential']:  # , 'step']:
                scheduler.step(epoch_num + (batch_num / num_batches_per_epoch))
            elif config['training']['scheduler']['name'] in ['cyclic']:
                scheduler.step()


        if show_pbar: pbar.close()

        # Calculate evaluation metric
        if mixup_is_enabled:
            train_mean_metric_value, train_metric_dict = metricHandler.calculate_mixup_metric(out_tot, targets_tot, mixup_lambda_epoch, mixup_indices_epoch)
        else:
            train_mean_metric_value, train_metric_dict = metricHandler.calculate_metric(out_tot, targets_tot)
            
        # Log epoch loss and AUC
        avg_loss = total_loss / num_batches_per_epoch
        logging.info(f'  Training   Loss={avg_loss:.5f}, {metric_name}s={train_metric_dict}')

        results_log.update({'train/loss':avg_loss})
        
        for key, val in train_metric_dict.items():
            results_log.update({f"train/{metric_name}_{key}" : val})
        
        results_log.update({f"train/mean_{metric_name}" : train_mean_metric_value})

        
        # Perform validation
        if epoch_num % config['training']['validation_interval'] == 0:
            val_loss, val_mean_metric_value, val_metric_dict, val_preds_dict, val_labels_dict, val_patientIDs_list = validate(config, model, loss_function, val_loader, metricHandler)
            # wandb.log({'train/loss': avg_loss, 'train/auc': avg_auc, 'val/loss': val_loss, 'val/auc': val_auc})
            logging.info(f'  Validation Loss={val_loss:.5f}, {metric_name}s={val_metric_dict}')
            results_log.update({'val/loss':val_loss})

            for key, val in val_metric_dict.items():
                results_log.update({f"val/{metric_name}_{key}" : val})
            results_log.update({f"val/mean_{metric_name}" : val_mean_metric_value})

            # check if the model has improved on this epoch
            best_value, improved = check_improvement(config, val_loss, val_mean_metric_value, best_value)
            
            if improved:
                patience_counter = 0
                if config['saving']['best_model']: 
                    save_model(config, model)
                else:
                    best_model_state_dict = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            # Check if patience has been exhausted
            if patience_counter >= config['training']['patience']:
                logging.info('Patience exhausted, stopping training')
                break
        
        logging.info(f'  Epoch duration = {(time.time() - start_epoch_time):.2f} seconds')
        

        if WandB_is_enabled(config):
            results_log.update({"epoch":epoch_num})
            wandb.log(results_log)

            if improved:
                best_log_dict = copy.deepcopy(results_log)
                update_WandB_summary_table(config, best_log_dict)

        
        # else:
        #     wandb.log({'train/loss': avg_loss, 'train/auc': avg_auc})

    # Log highest AUC
    # wandb.log({'train/highest_auc': highest_auc})
    # logging.info('Finished training')

    # Save the best model
    if config['saving']['best_model']: 
        model = load_model(config, model)
    else:
        model.load_state_dict(best_model_state_dict)    

    return model





