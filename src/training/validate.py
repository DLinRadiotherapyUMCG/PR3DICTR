import torch
import logging
import numpy as np 

from src.constants import DEVICE
from src.utils.move_batch_to_device import move_batch_to_device



def validate(config : dict, model, loss_function, val_loader, metric_handler, LabelTypesManager):
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

    total_loss = 0.0
    total_loss_dict = {lab: 0.0 for lab in labels}
    num_batches = 0
    

    sigmoid_act = torch.nn.Sigmoid()
    identity_act = torch.nn.Identity()
    
    patientIDs_list = []
    preds_dict = dict.fromkeys(labels, [])
    labels_dict = dict.fromkeys(labels, [])
    all_targets = torch.tensor([], device=DEVICE, dtype=torch.float32)  # Initialize an empty tensor for all targets
    
    for label in labels:
        preds_dict[label] = []
        labels_dict[label] = []

    # patient_IDs_list = val_loader.dataset.patient_IDs_list
    # print(patient_IDs_list)

    # for idx, ID in enumerate(patient_IDs_list):
    #     print(idx, ID)
    #     # get the data and target of the requested patient
    #     data = val_loader.dataset.__getitem__(idx)

    
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            
                logging.debug(f'Validation batch {i}')
                inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

                outputs = model(x=inputs, features=clinical_features)
                #mean_loss, loss_dict = loss_function(outputs, targets)

                #total_loss += mean_loss.item()       
                # for label in labels:
                #     total_loss_dict[label] += loss_dict[label].item()
                
                preds_dict, labels_dict = collect_all_preds_and_labels(labels, label_types, sigmoid_act, identity_act, preds_dict, labels_dict, targets, outputs)
                
                all_targets = torch.cat([all_targets, targets], dim=0) if len(all_targets) > 0 else targets

                
                num_batches += 1
                patientIDs_list += list(batch['patient_id'])
            
        avg_loss, total_loss_dict = loss_function(preds_dict, all_targets)

        # move all preds and labels to CPU ( NOTE: and convert to numpy arrays ?)
        for label in labels:
            preds_dict[label] = preds_dict[label].cpu().detach().numpy()
            labels_dict[label] = labels_dict[label].cpu().detach().numpy()



    mean_metric_value, metric_dict = metric_handler.calculate_metric(preds_dict, labels_dict)

    # avg_loss = total_loss / num_batches
    avg_loss = avg_loss.item()
    for label in labels:
        total_loss_dict[label] = total_loss_dict[label].item()

    logging.debug(f'Validation loss: {avg_loss}')

    return avg_loss, total_loss_dict, mean_metric_value, metric_dict, preds_dict, labels_dict, patientIDs_list



def collect_all_preds_and_labels(labels, label_types, sigmoid_act, identity_act, preds_dict, labels_dict, targets, outputs):

    for lab_idx, label in enumerate(labels):
        if label_types[lab_idx] == 'Binary':
            endpoint_preds = sigmoid_act(outputs[label])
            endpoint_labels = targets[:, lab_idx]
                    
        elif label_types[lab_idx] == 'Event':
                        # If the label is an event, we need to reshape the predictions and targets
            endpoint_preds = identity_act(outputs[label])
                        #endpoint_labels = targets[:, lab_idx:lab_idx+2]  Get both event and days labels                              # NOTE: temporarily commented out 
            endpoint_labels = targets[:, lab_idx]                                                # NOTE: TEMPORARILY CHANGED TO ONLY USE THE EVENT LABELS
                    
                    # collect all of the predictions and labels 
        preds_dict[label] = torch.cat(
                        [preds_dict[label], endpoint_preds], dim=0
                    ) if len(preds_dict[label]) > 0 else endpoint_preds
        labels_dict[label] = torch.cat(
                        [labels_dict[label], endpoint_labels], dim=0
                    ) if len(labels_dict[label]) > 0 else endpoint_labels
        
    return preds_dict, labels_dict