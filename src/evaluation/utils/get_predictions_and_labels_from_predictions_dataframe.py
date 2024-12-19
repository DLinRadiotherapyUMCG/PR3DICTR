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

    # get predictions of this set
    df_fold_preds = df_fold_all_preds[df_fold_all_preds["Mode"] == set_name]

    # collect all (valid) predictions and labels for each endpoint (i.e. mask missing values)
    endpoint_list = config['columns']['labels']
    predictions_per_endpoint_dict = {endpoint: None for endpoint in endpoint_list}
    labels_per_endpoint_dict = {endpoint: None for endpoint in endpoint_list}

    for endpoint in endpoint_list:
        pred = df_fold_preds[[f"{endpoint}_pred"]].to_numpy()
        true = df_fold_preds[[f"{endpoint}_true"]].to_numpy()

        # mask out the missing values
        true, pred = remove_missing(config, true, pred)

        # store in dict
        predictions_per_endpoint_dict[endpoint] = pred
        labels_per_endpoint_dict[endpoint] = true

    return predictions_per_endpoint_dict, labels_per_endpoint_dict



