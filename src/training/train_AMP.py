import os
import logging
import pandas as pd
import numpy as np
import time
import torch
import wandb
import copy
from tqdm import tqdm
from torch.amp import GradScaler

from src.constants import DEVICE
from src.utils.loss_func.get_loss_function import get_loss_function
from src.utils.optimizer.get_optimizer import get_optimizer
from src.utils.scheduler.get_scheduler import get_scheduler
from src.utils.move_batch_to_device import move_batch_to_device
from src.training.validate import validate
from src.models.tools.save_model import save_model, load_model
from src.training.utils.check_improvement import check_improvement

from src.hyper_opt.WandB_functions import is_WandB_enabled, update_WandB_summary_table, WandB_log
from src.visualization.plot_model_inputs import plot_model_inputs
from src.evaluation.mainMetricHandler import mainMetricHandler
from src.utils.loss_func.calc_mixup_loss import calc_mixup_loss
from src.training.utils.gradNorm import GradNorm
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
    if config['training']['GradNorm']['isEnabled']:
        gradNorm = GradNorm(config = config,
                            model = model, # model.output_head.linear_layers.shared_fc_layers, 
                            alpha = config['training']['GradNorm']['alpha'], 
                            lr = config['training']['GradNorm']['learning_rate'], 
                            WandB_is_enabled = config['hyperparam_tuning']['WandB']['isEnabled'])



    # Initialize the best model and lowest validation loss
    best_model_state_dict = None
    if(config['training']['stopping_criteria'] == "loss"):
        print("Optimizing based on loss")
        best_value = np.inf
    else:
        best_value = 0
        
    patience_counter = 0
    
    scaler = GradScaler()

    sigmoid_act = torch.nn.Sigmoid()


    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    


    # Training loop
    logging.info('Starting training loop')
    for epoch_num in range(1, config['training']['max_epochs'] + 1):
        mixup_lambda_epoch = None
        #current_epoch_num = epoch+1
        improved = False # Flag to indicate if the model has improved on this epoch
        results_log = dict()  # results log for the current epoch (for WandB)
        best_log_dict = None  # results dict of the best epoch thus far
        out_tot = dict.fromkeys(labels, [])
        targets_tot = dict.fromkeys(labels, [])
        
        
        # for label in labels: # TODO: check that this can be deleted, and that the lines above work
        #     out_tot[label] = []
        #     targets_tot[label] = []

        logging.info(f'Epoch {epoch_num}')
        model.train()

        total_loss = 0.0
        train_loss_dict = {label : 0 for label in labels}
        num_batches_per_epoch = len(train_loader)

        start_epoch_time = time.time()
  
        if show_pbar:
            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch_num}', position=0, leave=True)

        all_batch_retrieval_times = []
        all_forward_pass_times = []
        all_backward_pass_times = []
        all_compute_loss_times = []
        all_dataloader_times = []
        all_optimizer_step_times = []

        dataloader_start_time = time.time()

        for batch_num, batch in enumerate(train_loader, start=1):  # , disable=config["hyperparam_tuning"]["optuna"]["IsEnabled"]
            dataloader_end_time = time.time()
            dataloader_time = (dataloader_end_time - dataloader_start_time)   # Convert to milliseconds
            all_dataloader_times.append(dataloader_time)
            
            # Record the start event
            #start_event.record()
            move_to_device_start_time = time.time()
            #logging.debug(f'Batch {batch_num} of epoch {epoch_num}')

            if show_pbar: pbar.update(1)

            optimizer.zero_grad(set_to_none=True)
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

            # Record the end event
            #end_event.record()
            move_to_device_end_time = time.time()
            #torch.cuda.synchronize()  # Wait for the events to complete and calculate elapsed time
            batch_retrieval_time = move_to_device_end_time - move_to_device_start_time  # Time in milliseconds
            all_batch_retrieval_times.append(batch_retrieval_time)
            logging.debug(f'Time to retrieve batch {batch_num}: {batch_retrieval_time:.4f} ms')
            
            # if mixup is enabled, then we need to know the lambda and indices for this batch, to compute the loss properly
            if mixup_is_enabled:
                mixup_lambda = batch['lambda']
                mixup_indices_batch = batch['indices']
                
                cur_batch_size = inputs.shape[0]
                mixup_lambda_batch = [mixup_lambda] * cur_batch_size
                mixup_adjusted_indices_batch = (batch_num-1) * cur_batch_size + mixup_indices_batch

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
            #start_event.record()
            with torch.autocast(device_type='cuda', dtype=torch.float16):

                forward_pass_start_time = time.time()
                outputs = model(x=inputs, features=clinical_features)

                # Record the end event
                #end_event.record()
                #torch.cuda.synchronize()  # Wait for the events to complete and calculate elapsed time
                #forward_pass_time = start_event.elapsed_time(end_event)  # Time in milliseconds
                forward_pass_time = time.time() - forward_pass_start_time
                all_forward_pass_times.append(forward_pass_time)
                logging.debug(f'Time to forward pass {batch_num}: {forward_pass_time:.4f} ms')

                # Record the start event
                #start_event.record()
                compute_loss_start_time = time.time()

                # Calculate loss
                if mixup_is_enabled:
                    # the loss formula is different for mixup
                    loss, loss_dict = calc_mixup_loss(outputs, targets, loss_function, mixup_indices_batch, mixup_lambda)                
                else:
                    # normal loss calculation
                    loss, loss_dict = loss_function(outputs, targets) 

                # Record the end event
                # end_event.record()
                # torch.cuda.synchronize()  # Wait for the events to complete and calculate elapsed time
                # compute_loss_time = start_event.elapsed_time(end_event)  # Time in milliseconds
                compute_loss_time = time.time() - compute_loss_start_time
                all_compute_loss_times.append(compute_loss_time)
                logging.debug(f'Time to compute loss {batch_num}: {compute_loss_time:.4f} ms')


            if config['training']['GradNorm']['isEnabled']:
                # reweight the losses using GradNorm (the loss is backpropagated in the GradNorm class)
                start_backward_pass_time = time.time()
                optimizer.zero_grad(set_to_none=True)
                loss = torch.stack([loss_dict[label] for label in labels])

                #start_event.record()

                #loss = gradNorm.step(loss)
                loss = gradNorm.step(loss, scaler=scaler)

                # Record the end event
                # end_event.record()
                # torch.cuda.synchronize()  # Wait for the events to complete and calculate elapsed time
                # backward_pass_time = start_event.elapsed_time(end_event)  # Time in milliseconds
                backward_pass_time = time.time() - start_backward_pass_time
                all_backward_pass_times.append(backward_pass_time)
                logging.debug(f'Time to backward pass {batch_num}: {backward_pass_time:.4f} ms')
            else:
                # backpropagate the loss normally
                if loss.grad_fn:
                    #start_event.record()
                    start_backward_pass_time = time.time()

                    #loss.backward()
                    scaler.scale(loss).backward()

                    # Record the end event
                    # end_event.record()
                    # torch.cuda.synchronize()  # Wait for the events to complete and calculate elapsed time
                    # backward_pass_time = start_event.elapsed_time(end_event)  # Time in milliseconds
                    backward_pass_time = time.time() - start_backward_pass_time
                    all_backward_pass_times.append(backward_pass_time)
                    logging.debug(f'Time to backward pass {batch_num}: {backward_pass_time:.4f} ms')

            # Calculate AUC            
            for idx, label in enumerate(labels):
                out_tot[label] = out_tot[label] + list(sigmoid_act(outputs[label]).cpu().detach().numpy().reshape((1,targets[:,idx].shape[0]))[0])
                targets_tot[label] = targets_tot[label] + list(targets[:,idx].cpu().detach().numpy().reshape((1,targets[:,idx].shape[0]))[0])

            # Log the batch loss to the epoch loss
            total_loss += loss.item()
            for label in labels:
                train_loss_dict[label] += loss_dict[label].item()

            # Update model weights
            
            #optimizer.step()
            if batch_num % 4 == 0 or batch_num == num_batches_per_epoch:
                #start_event.record()
                start_optimise_time = time.time()

                scaler.step(optimizer)
                
            
                scaler.update()

                optimizer.zero_grad(set_to_none=True)

                # Record the end event
                # end_event.record()
                # torch.cuda.synchronize()  # Wait for the events to complete and calculate elapsed time
                # optimizer_step_time = start_event.elapsed_time(end_event)  # Time in milliseconds
                optimizer_step_time = time.time() - start_optimise_time
                all_optimizer_step_times.append(optimizer_step_time)
                logging.debug(f'Time to optimizer step {batch_num}: {optimizer_step_time:.4f} ms')

            # Step the scheduler
            if config['training']['scheduler']['name'] in ['cosine', 'exponential']:  # , 'step']:
                scheduler.step(epoch_num + (batch_num / num_batches_per_epoch))
            elif config['training']['scheduler']['name'] in ['cyclic']:
                scheduler.step()


            # TIMING
            dataloader_start_time = time.time()

        print(f"Total batch retrieval time: {np.sum(all_batch_retrieval_times):.4f} s")
        print(f"Total forward pass time: {np.sum(all_forward_pass_times):.4f} s")
        print(f"Total backward pass time: {np.sum(all_backward_pass_times):.4f} s")
        print(f"Total compute loss time: {np.sum(all_compute_loss_times):.4f} s")
        print(f"Total dataloader time: {np.sum(all_dataloader_times):.4f} s")
        print(f"Total optimizer step time: {np.sum(all_optimizer_step_times):.4f} s")
        

        if show_pbar: pbar.close()

        # Calculate evaluation metric
        if mixup_is_enabled:
            train_mean_metric_value, train_metric_dict = metricHandler.calculate_mixup_metric(out_tot, targets_tot, mixup_lambda_epoch, mixup_indices_epoch)
        else:
            train_mean_metric_value, train_metric_dict = metricHandler.calculate_metric(out_tot, targets_tot)
            
        # Log epoch loss and AUC
        avg_loss = total_loss / num_batches_per_epoch
        for label in labels:
            train_loss_dict[label] = train_loss_dict[label] / num_batches_per_epoch
                
        logging.info(f'  Training   Loss={avg_loss:.5f}, {metric_name}s={train_metric_dict}')

        results_log.update({'loss_train/mean_loss':avg_loss})
        
        for key, val in train_metric_dict.items():
            results_log.update({f"train/{key}_{metric_name}" : val})
            results_log.update({f"loss_train/{key}" : train_loss_dict[key]})
        
        results_log.update({f"train/mean_{metric_name}" : train_mean_metric_value})
        
        # Perform validation
        if epoch_num % config['training']['validation_interval'] == 0:

            start_val_time = time.time()
            val_loss_value, val_loss_dict, val_mean_metric_value, val_metric_dict, val_preds_dict, val_labels_dict, val_patientIDs_list = validate(config, model, loss_function, val_loader, metricHandler)
            logging.info(f'  Validation Loss={val_loss_value:.5f}, {metric_name}s={val_metric_dict}')
            #results_log.update({'val/loss':val_loss_value})

            for key, val in val_metric_dict.items():
                results_log.update({f"val/{key}_{metric_name}" : val})
                results_log.update({f"loss_val/{key}" : val_loss_dict[key]})

            results_log.update({f"val/mean_{metric_name}" : val_mean_metric_value})
            results_log.update({f"loss_val/mean_loss" : val_loss_value})

            logging.info(f'  Validation duration = {(time.time() - start_val_time):.2f} seconds')

            # check if the model has improved on this epoch
            best_value, improved = check_improvement(config, val_loss_value, val_mean_metric_value, best_value)
            
            # if it has improved, then save the model and reset the counter
            if improved:
                patience_counter = 0
                if config['saving']['best_model']: 
                    save_model(config, model)
                else:
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                
                best_log_dict = copy.deepcopy(results_log)
                update_WandB_summary_table(config, best_log_dict)
            else:
                patience_counter += 1

            # Check if patience has been exhausted
            if patience_counter >= config['training']['patience']:
                logging.info('Patience exhausted, stopping training')
                break
        
        logging.info(f'  Epoch duration = {(time.time() - start_epoch_time):.2f} seconds')
        
        # log this epoch's results to WandB
        WandB_log(config, results_log, epoch = epoch_num)               


    # Load the best model, and return it
    if config['saving']['best_model']: 
        model = load_model(config, model)
    else:
        model.load_state_dict(best_model_state_dict)    

    return model





