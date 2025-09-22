


def make_mean_predictions_dataframe(df_all_test_preds, endpoint_list):
    """
    Create a DataFrame with the mean predictions for each toxicity endpoint.
    
    Args:
        df_all_test_preds (pd.DataFrame): DataFrame containing all model predictions.
        endpoint_list (list): List of toxicity endpoints.
        
    Returns:
        pd.DataFrame: DataFrame with true labels and mean predictions for each endpoint.
    """

    true_label_columns = [col for col in df_all_test_preds if col.endswith("_true")]  # the columns with true labels

   # make a file with the true labels and mean predictions for each toxicity
    df_mean_predictions = df_all_test_preds[true_label_columns].copy()  # start with the true labels
    df_mean_predictions.index.name = 'PatientID'  # set the index name to Patient

    for endpoint in endpoint_list:
        all_endpoint_prediction_columns = [col for col in df_all_test_preds.columns if endpoint in col and 'pred' in col]
        mean_endpoint_prediction = df_all_test_preds[all_endpoint_prediction_columns].mean(axis=1)
        #print(f"Mean predictions for {endpoint}:", mean_prediction.head())
        df_mean_predictions[f"{endpoint}_mean_pred"] = mean_endpoint_prediction

    return df_mean_predictions