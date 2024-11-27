import pandas as pd
import torch
import os

def concatenate_predictions(config, list_of_pred_list_dicts, list_of_true_list_dicts):
    """
    
    Args:
        Six dicts of lists with the predictions and true labels for each train, val, test set.
    Returns:
        all_y_pred_list_dict: a dict of lists with all of the predictions for each endpoint
        all_y_true_list_dict: a dict of lists with all of the true labels for each endpoint
    """
    endpoint_list = config['columns']['label']

    all_y_pred_list_dict = {endpoint: [] for endpoint in endpoint_list}
    all_y_true_list_dict = {endpoint: [] for endpoint in endpoint_list}

    for [preds, labels] in zip(list_of_pred_list_dicts, list_of_true_list_dicts):
        for endpoint in endpoint_list:
            all_y_pred_list_dict[endpoint] += preds[endpoint]
            all_y_true_list_dict[endpoint] += labels[endpoint]
       
    return all_y_pred_list_dict, all_y_true_list_dict



def save_predictions(config, patient_ids, y_pred_list_dict, y_true_list_dict, mode_list, external_set=False):
    """
    Save prediction and corresponding true labels to csv.

    Args:
        patient_ids (list): list of patient ids
        endpoint_list (list): list of endpoints
        y_pred_list_dict (dict of lists): dict of lists of PyTorch tensors
        y_true_list_dict (dict of lists): dict of lists of PyTorch tensors
        mode_list (list): list of strings
        model_name (str):
        exp_dir (str):
        logger:

    Returns:

    """

    endpoint_list = config['columns']['label']

    # Initialize df
    df_patient_ids = pd.DataFrame(patient_ids, columns=['PatientID'])
    df_mode = pd.DataFrame(mode_list, columns=['Mode'])
    df_y = pd.concat([df_patient_ids, df_mode], axis=1)

    for endpoint in endpoint_list:
        # Convert to CPU
        y_pred = y_pred_list_dict[endpoint]
        y_true = y_true_list_dict[endpoint]

        # Save to DataFrame
        if config['model']['num_classes'] == 1:
            df_y_pred = pd.DataFrame(y_pred, columns=['{}_pred'.format(endpoint)])
            df_y_true = pd.DataFrame(y_true, columns=['{}_true'.format(endpoint)])
        else:
            y_pred = torch.tensor(y_pred)
            y_true = torch.tensor(y_true)
            num_cols = y_pred.shape[1]            
            df_y_pred = pd.DataFrame(y_pred, columns=['{}_pred_{}'.format(endpoint, c) for c in range(num_cols)])
            df_y_true = pd.DataFrame(y_true, columns=['{}_true_{}'.format(endpoint, c) for c in range(num_cols)])
        df_y = pd.concat([df_y, df_y_pred, df_y_true], axis=1)

    # Save to file
    if external_set:
        #print(model_name)
        output_filename = os.path.join(config['general']['resultsCurrentDirectory'], "model_predictions.csv")
    else:
        output_filename = os.path.join(config['general']['resultsCurrentDirectory'], "model_predictions.csv")

    df_y.to_csv(output_filename, sep=';', index=False)

