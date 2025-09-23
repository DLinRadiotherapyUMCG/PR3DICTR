import pandas as pd

from src.evaluation.metrics.utils import remove_missing


def get_predictions_and_labels_from_predictions_dataframe(config : dict, df_fold_all_preds: pd.DataFrame, set_name:str):
    """
    This function extracts the predictions and labels from the predictions.csv dataframe (from each fold) for a specific set_name (train, val, or test). It also drops the missing values.

    Args:
        config (dict): config dictionary
        df_fold_all_preds (pd.DataFrame): dataframe with predictions and labels
        set_name (str): name of the set (train, val, or test)
    Returns:
        predictions_per_endpoint_dict (dict): dict with predictions for each endpoint
        labels_per_endpoint_dict (dict): dict with labels for each endpoint
    """
    all_label_column_names = config['saving']['label_column_names'] # .label_names_full_list


    # get predictions of this set
    df_fold_preds = df_fold_all_preds[df_fold_all_preds["Mode"] == set_name]

    # collect all (valid) predictions and labels for each endpoint (i.e. mask missing values)
    endpoint_list = config['columns']['labels']
    predictions_per_endpoint_dict = {endpoint: None for endpoint in endpoint_list}
    labels_per_endpoint_dict = {endpoint: None for endpoint in endpoint_list}

    for endpoint, label_column_names in zip(endpoint_list, all_label_column_names):
        pred = df_fold_preds[[f"{endpoint}_pred"]].to_numpy()

        label_column_names = [label_column_names] if isinstance(label_column_names, str) else label_column_names
        label_columns = ['{}_true'.format(x) for x in label_column_names]
        true = df_fold_preds[label_columns].to_numpy()

        # mask out the missing values
        true, pred = remove_missing(config, true, pred)

        # store in dict
        predictions_per_endpoint_dict[endpoint] = pred
        labels_per_endpoint_dict[endpoint] = true

    return predictions_per_endpoint_dict, labels_per_endpoint_dict



