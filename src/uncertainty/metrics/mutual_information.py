import numpy as np

from src.uncertainty.metrics.entropy import binary_entropy


def mutual_information(predictions):    
    """
    Calculate the mutual information of the predictions.
    
    Args:
        predictions (pd.Dataframe): Dataframe containing the predictions from multiple models.
    Returns:
        pd.Series: Series containing the mutual information of the predictions for each patient.
    """
    # Calculate the mean
    mean_predictions = predictions.mean(axis=1)
    
    entropy_of_mean = binary_entropy(mean_predictions)
    entropy_per_pass = binary_entropy(predictions)

    mean_entropy = np.mean(entropy_per_pass, axis=1)

    # Calculate mutual information
    mutual_info = entropy_of_mean - mean_entropy

    return mutual_info




