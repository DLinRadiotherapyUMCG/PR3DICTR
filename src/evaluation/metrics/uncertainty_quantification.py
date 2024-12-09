import numpy as np


def calculate_predictive_variance(predictions):
    """
    Calculate the predictive variance of the predictions.
    :param predictions: DataFrame containing the predictions of the models.
    :return: Series containing the predictive variance.
    """
    return None


def calculate_predictive_entropy(predictions):
    """
    Calculate the predictive entropy of the predictions.
    :param predictions: DataFrame containing the predictions of the models.
    :return: Series containing the predictive entropy.
    """
    mean = predictions.mean(axis=1)
    entropy = - (mean * np.log(mean + 1e-10) + (1 - mean) * np.log(1 - mean + 1e-10))
    return np.round(entropy, 4)

def calculate_aleatoric_uncertainty(predictions):
    """
    Calculate the aleatoric uncertainty of the predictions.

    Aleatoric uncertainty captures the uncertainty inherent in the data.
    Estimated by the variance of the predictions.

    :param predictions: DataFrame containing the predictions of the models.
    :return: Series containing the aleatoric uncertainty.
    """
    aleatoric_uncertainty = np.mean(predictions * (1 - predictions), axis=1)
    return aleatoric_uncertainty

def calculate_epistemic_uncertainty(predictions):
    """
    Calculate the epistemic uncertainty of the predictions.

    Epistemic uncertainty captures the uncertainty due in the model parameters.
    Estimated by the variance of mean predictions

    :param predictions: DataFrame containing the predictions of the models.
    :return: Series containing the epistemic uncertainty.
    """

    # Only if not already a numpy array
    if not isinstance(predictions, np.ndarray):
        predictions_np = predictions.to_numpy()  # Convert DataFrame to numpy array
    else:
        predictions_np = predictions

    return np.mean(predictions_np ** 2, axis=1) - np.mean(predictions_np, axis=1) ** 2
