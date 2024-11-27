import torch
import logging
from src.constants import DEVICE

from src.training.tools.utils import move_batch_to_device
from src.evaluation.calculate_auc import calculate_auc_multi


def validate(config, model, loss_function, val_loader):
    model.eval()

    total_loss = 0.0
    total_auc = 0.0
    num_batches = 0
    #num_auc_batches = config['training']['validation']['num_auc_batches']
    labels = config['columns']['label']
    
    
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
                preds_dict[label] = preds_dict[label] + list(outputs[label].cpu().detach().numpy().reshape((1,targets[:,lab_indx].shape[0]))[0])
                labels_dict[label] = labels_dict[label] + list(targets[:,lab_indx].cpu().detach().numpy().reshape((1,targets[:,lab_indx].shape[0]))[0])
            
            num_batches += 1
            patientIDs_list += list(batch['patient_id'])

    auc_dict = calculate_auc_multi(preds_dict, labels_dict, config)
    model.train()

    avg_loss = total_loss / num_batches

    logging.debug(f'Validation loss: {avg_loss}')
    # logging.debug(f'Validation AUC: {avg_auc}')

    return avg_loss, auc_dict, preds_dict, labels_dict, patientIDs_list