import torch
import logging
import numpy as np 

from src.constants import DEVICE
from src.training.tools.utils import move_batch_to_device



def validate(config, model, loss_function, val_loader, metric_handler):
    """
    Evaluate the model on the given dataloader.
    
    Returns:
      avg_loss: average loss over the dataloader
      auc_dict: dictionary of AUC values for each label
      preds_dict: dictionary of predictions for each label
      labels_dict: dictionary of true labels for each label
      patientIDs_list: list of patient IDs in the dataloader
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0
    labels = config['columns']['labels']

    sigmoid_act = torch.nn.Sigmoid()
    
    patientIDs_list = []
    preds_dict = dict.fromkeys(labels)
    labels_dict = dict.fromkeys(labels)
    
    for label in labels:
        preds_dict[label] = []
        labels_dict[label] = []

    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            logging.debug(f'Validation batch {i}')
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

            outputs = model(x=inputs, features=clinical_features)
            loss = loss_function(outputs, targets)

            total_loss += loss.item()            
            
            for lab_indx, label in enumerate(labels):
                preds_dict[label] = preds_dict[label] + list(sigmoid_act(outputs[label]).cpu().detach().numpy().reshape((1,targets[:,lab_indx].shape[0]))[0])
                labels_dict[label] = labels_dict[label] + list(targets[:,lab_indx].cpu().detach().numpy().reshape((1,targets[:,lab_indx].shape[0]))[0])
            
            num_batches += 1
            patientIDs_list += list(batch['patient_id'])

    mean_metric_value, metric_dict = metric_handler.calculate_metric(preds_dict, labels_dict)
    #auc_dict = calculate_auc_multi(preds_dict, labels_dict, config)
    model.train()

    avg_loss = total_loss / num_batches

    logging.debug(f'Validation loss: {avg_loss}')
    # logging.debug(f'Validation AUC: {avg_auc}')

    return avg_loss, mean_metric_value, metric_dict, preds_dict, labels_dict, patientIDs_list