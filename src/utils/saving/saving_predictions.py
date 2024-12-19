import pandas as pd
import torch
import os

from src.utils.saving.get_predictions_csv_dir import get_predictions_csv_dir


def concatenate_predictions(config, list_of_pred_list_dicts: list[dict], list_of_true_list_dicts: list[dict]):
    """
    Function to concatenate N list dicts of predictions and true labels into a single list dict of preds and true labels.
    For example, this function can merge together all of the training preds and validation preds into a single dict of preds.
    
    Args:
        config (dict): config dictionary
        list_of_pred_list_dicts (list of dicts): list of dicts of lists of PyTorch tensors
        list_of_true_list_dicts (list of dicts): list of dicts of lists of PyTorch tensors
    Returns:
        all_y_pred_list_dict (dict): a dict of lists with all of the predictions for each endpoint
        all_y_true_list_dict (dict): a dict of lists with all of the true labels for each endpoint
    """
    endpoint_list = config['columns']['labels']

    all_y_pred_list_dict = {endpoint: [] for endpoint in endpoint_list}
    all_y_true_list_dict = {endpoint: [] for endpoint in endpoint_list}

    for [preds, labels] in zip(list_of_pred_list_dicts, list_of_true_list_dicts):
        for endpoint in endpoint_list:
            all_y_pred_list_dict[endpoint] += preds[endpoint]
            all_y_true_list_dict[endpoint] += labels[endpoint]
       
    return all_y_pred_list_dict, all_y_true_list_dict



def save_predictions(config: dict, patient_ids: list[str], y_pred_list_dict: dict, y_true_list_dict: dict, mode_list: list[str], 
                     is_test_set:bool=False, ensemble_predictions: bool = False):

    """
    Save prediction and corresponding true labels to csv.

    Args:
        config (dict): config dictionary
        patient_ids (list): list of patient IDs (each ID is a string)
        y_pred_list_dict (dict of lists): dict of lists of PyTorch tensors
        y_true_list_dict (dict of lists): dict of lists of PyTorch tensors
        mode_list (list): list of strings (e.b. ['train', 'val', 'train', 'train','train','val', 'train', ...])

    Returns:

    """

    endpoint_list = config['columns']['labels']

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
    output_file_dir = get_predictions_csv_dir(config, is_test_set, ensemble_predictions)

    df_y.to_csv(output_file_dir, sep=';', index=False)

