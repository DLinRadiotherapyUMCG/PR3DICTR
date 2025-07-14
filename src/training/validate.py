import torch
import logging
import numpy as np 

from src.constants import DEVICE
from src.utils.move_batch_to_device import move_batch_to_device
from src.training.utils.collect_all_preds_and_labels import collect_all_preds_and_labels


def validate(config : dict, model, loss_function, val_loader, metric_handler):
    """
    Evaluate the model on the given dataloader.
    Args:
        config (dict): config params
        model: the model to validate
        loss_function:
        val_loader: dataloader with the dataset to validate the model on
        metric_handler: class object to compute the main metric specified in the config
    Returns:
        avg_loss: average loss over the dataloader
        total_loss_dict: average loss per label
        mean_metric_value: mean metric value
        metric_dict: dictionary of metric values for each label
        preds_dict: dictionary of predictions for each label
        labels_dict: dictionary of true labels for each label
        patientIDs_list: list of patient IDs in the dataloader
    """
    labels = config['columns']['labels']
    label_types = config['columns']['labels_types']

    model.eval()
    
    # to collect all of the predictions and labels
    patientIDs_list = []
    preds_dict = dict.fromkeys(labels, [])
    labels_dict = dict.fromkeys(labels, [])
    all_targets = torch.tensor([], device=DEVICE, dtype=torch.float32)  # also save all of the targets in a single tensor (to calculate the loss at the end)
    
    for label in labels:
        preds_dict[label] = []
        labels_dict[label] = []
    
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
                logging.debug(f'Validation batch {i}')
                inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

                outputs = model(x=inputs, features=clinical_features)
                
                # collect all predictions and labels for each label type
                preds_dict, labels_dict = collect_all_preds_and_labels(labels, label_types, preds_dict, labels_dict, targets, outputs)
                
                all_targets = torch.cat([all_targets, targets], dim=0) if len(all_targets) > 0 else targets
                patientIDs_list += list(batch['patient_id'])
        
    # for validation, we can calculate the loss just once at the end (instead of for each batch)
    avg_loss, total_loss_dict = loss_function(preds_dict, all_targets)
    # convert the losses from tensors to floats
    avg_loss = avg_loss.item()
    for label in labels:
        total_loss_dict[label] = total_loss_dict[label].item()
    logging.debug(f'Validation loss: {avg_loss}')

    # move all preds and labels to CPU
    for label in labels:
        preds_dict[label] = preds_dict[label].cpu().detach().numpy()
        labels_dict[label] = labels_dict[label].cpu().detach().numpy()

    mean_metric_value, metric_dict = metric_handler.calculate_metric(preds_dict, labels_dict)

    # avg_loss = total_loss / num_batches

    return avg_loss, total_loss_dict, mean_metric_value, metric_dict, preds_dict, labels_dict, patientIDs_list


