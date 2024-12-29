import torch
import logging
import numpy as np 

from src.constants import DEVICE
from src.utils.move_batch_to_device import move_batch_to_device



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

    model.eval()

    total_loss = 0.0
    total_loss_dict = {lab: 0.0 for lab in labels}
    num_batches = 0
    

    sigmoid_act = torch.nn.Sigmoid()
    
    patientIDs_list = []
    preds_dict = dict.fromkeys(labels, [])
    labels_dict = dict.fromkeys(labels, [])
    
    for label in labels:
        preds_dict[label] = []
        labels_dict[label] = []

    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            logging.debug(f'Validation batch {i}')
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

            outputs = model(x=inputs, features=clinical_features)
            mean_loss, loss_dict = loss_function(outputs, targets)

            total_loss += mean_loss.item()       
            for label in labels:
                total_loss_dict[label] += loss_dict[label].item()
            
            for lab_idx, label in enumerate(labels):
                preds_dict[label] = preds_dict[label] + list(sigmoid_act(outputs[label]).cpu().detach().numpy().reshape((1,targets[:,lab_idx].shape[0]))[0])
                labels_dict[label] = labels_dict[label] + list(targets[:,lab_idx].cpu().detach().numpy().reshape((1,targets[:,lab_idx].shape[0]))[0])
            
            num_batches += 1
            patientIDs_list += list(batch['patient_id'])

    mean_metric_value, metric_dict = metric_handler.calculate_metric(preds_dict, labels_dict)

    avg_loss = total_loss / num_batches
    for label in labels:
        total_loss_dict[label] = total_loss_dict[label] / num_batches

    logging.debug(f'Validation loss: {avg_loss}')

    return avg_loss, total_loss_dict, mean_metric_value, metric_dict, preds_dict, labels_dict, patientIDs_list