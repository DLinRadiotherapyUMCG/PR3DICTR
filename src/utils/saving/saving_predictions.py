import pandas as pd
import torch
import os

from src.utils.saving.get_predictions_csv_dir import get_predictions_csv_dir
import numpy as np

def concatenate_predictions(config, list_of_pred_dicts: list[dict], list_of_true_dicts: list[dict]):
    """
    Function to concatenate N dicts of numpy arrays of predictions and true labels into a single dict of numpy arrays.
    For example, this function can merge together all of the training preds and validation preds into a single dict of preds.
    
    Args:
        config (dict): config dictionary
        list_of_pred_dicts (list of dicts): list of dicts of numpy arrays
        list_of_true_dicts (list of dicts): list of dicts of numpy arrays
    Returns:
        all_y_pred_dict (dict): a dict of numpy arrays with all of the predictions for each endpoint
        all_y_true_dict (dict): a dict of numpy arrays with all of the true labels for each endpoint
    """
    endpoint_list = config['columns']['labels']

    all_y_pred_dict = {}
    all_y_true_dict = {}

    for endpoint in endpoint_list:

        preds = [pred_dict[endpoint] for pred_dict in list_of_pred_dicts if pred_dict[endpoint] is not None]
        trues = [true_dict[endpoint] for true_dict in list_of_true_dicts if true_dict[endpoint] is not None]

        #print(endpoint, "concat shapes", preds.shape, trues.shape)
        all_y_pred_dict[endpoint] = np.concatenate(preds, axis=0)
        all_y_true_dict[endpoint] = np.concatenate(trues, axis=0)

    return all_y_pred_dict, all_y_true_dict

def save_predictions(config: dict, LabelTypesManager, patient_ids: list[str], y_pred_list_dict: dict, y_true_list_dict: dict, mode_list: list[str], 
                     is_test_set:bool=False, ensemble_predictions: bool = False, filename: str = None):

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

    all_label_column_names = config['saving']['label_column_names'] 

    # Initialize df
    df_patient_ids = pd.DataFrame(patient_ids, columns=['PatientID'])
    df_mode = pd.DataFrame(mode_list, columns=['Mode'])
    df_y = pd.concat([df_patient_ids, df_mode], axis=1)

    for endpoint, label_column_names in zip(endpoint_list, all_label_column_names):
        # Convert to CPU
        y_pred = y_pred_list_dict[endpoint]
        y_true = y_true_list_dict[endpoint]

        # Save to DataFrame
        if config['model']['output_head']['num_ohe_classes'] == 1:
            df_y_pred = pd.DataFrame(y_pred, columns=['{}_pred'.format(endpoint)])

            # check if this endpoint has just one column in the label, or multiple columns (e.g. event and days)
            label_column_names = [label_column_names] if isinstance(label_column_names, str) else label_column_names
            label_columns = ['{}_true'.format(x) for x in label_column_names]
            df_y_true = pd.DataFrame(y_true, columns=label_columns)
        else:
            y_pred = torch.tensor(y_pred)
            y_true = torch.tensor(y_true)
            num_cols = y_pred.shape[1]            
            df_y_pred = pd.DataFrame(y_pred, columns=['{}_pred_{}'.format(endpoint, c) for c in range(num_cols)])
            df_y_true = pd.DataFrame(y_true, columns=['{}_true_{}'.format(endpoint, c) for c in range(num_cols)])
        df_y = pd.concat([df_y, df_y_pred, df_y_true], axis=1)

    # Save to file
    output_file_dir = get_predictions_csv_dir(config, is_test_set, ensemble_predictions, filename=filename)

    df_y.to_csv(output_file_dir, sep=';', index=False)

