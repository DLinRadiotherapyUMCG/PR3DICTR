import torch
import logging
import numpy as np 
import pandas as pd

from src.constants import DEVICE
from src.utils.move_batch_to_device import move_batch_to_device
from src.uncertainty.utils.enable_dropout import enable_dropout


def collect_one_predictions_pass(config : dict, model, data_loader, enable_MC_dropout=True):                                # TODO: Test time augmentation with this same function ????
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
    endpoint_list = config['columns']['labels']

    model.eval()
    if enable_MC_dropout:
        enable_dropout(model)

    sigmoid_act = torch.nn.Sigmoid()
    
    patientIDs_list = []
    preds_dict = dict.fromkeys(endpoint_list, [])
    labels_dict = dict.fromkeys(endpoint_list, [])
    
    for label in endpoint_list:
        preds_dict[label] = []
        labels_dict[label] = []
    
    # collect the predictions and labels
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            
            logging.debug(f'Validation batch {i}')
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

            outputs = model(x=inputs, features=clinical_features)
            
            for lab_idx, label in enumerate(endpoint_list):
                preds_dict[label] = preds_dict[label] + list(sigmoid_act(outputs[label]).cpu().detach().numpy().reshape((1,targets[:,lab_idx].shape[0]))[0])
                labels_dict[label] = labels_dict[label] + list(targets[:,lab_idx].cpu().detach().numpy().reshape((1,targets[:,lab_idx].shape[0]))[0])
            
            patientIDs_list += list(batch['patient_id'])


    # merge the predictions and labels into a DataFrame
    df_preds = pd.DataFrame(patientIDs_list, columns=['PatientID'])
    for endpoint in endpoint_list:
        # Convert to CPU
        y_pred = preds_dict[endpoint]
        y_true = labels_dict[endpoint]

        # Save to DataFrame
        if config['model']['num_ohe_classes'] == 1:
            df_y_pred = pd.DataFrame(y_pred, columns=['{}_pred'.format(endpoint)])
            df_y_true = pd.DataFrame(y_true, columns=['{}_true'.format(endpoint)])
        else:
            y_pred = torch.tensor(y_pred)
            y_true = torch.tensor(y_true)
            num_cols = y_pred.shape[1]            
            df_y_pred = pd.DataFrame(y_pred, columns=['{}_pred_{}'.format(endpoint, c) for c in range(num_cols)])
            df_y_true = pd.DataFrame(y_true, columns=['{}_true_{}'.format(endpoint, c) for c in range(num_cols)])
        df_preds = pd.concat([df_preds, df_y_pred, df_y_true], axis=1)
    
    return df_preds






