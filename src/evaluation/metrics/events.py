import numpy as np
from src.evaluation.metrics.utils import remove_missing

def c_index(config, labels, preds, sample_weights=None):

    labels, preds = remove_missing(config,labels,preds) # removes missing values 

    events, days = labels[:, 0], labels[:, 1]  # unpack the events and days from the labels

    # Compute the C index
    c_index_value = compute_concordance_index(
        event_times=days,
        predicted_scores=preds,
        event_observed=events
    )

    return c_index_value

def compute_concordance_index(event_times, predicted_scores, event_observed):
    """
    Compute the concordance index (C-index) for survival models.

    Parameters:
        event_times (array-like): Actual survival times.
        predicted_scores (array-like): Predicted risk scores (higher = more risk).
        event_observed (array-like): Event occurred (1) or censored (0).

    Returns:
        float: Concordance index.
    """
    event_times = np.asarray(event_times)
    predicted_scores = np.asarray(predicted_scores)
    event_observed = np.asarray(event_observed)

    # For calculating the C-index we will compare all patients i,j  
    n = 0 # Total number of compareable samples
    n_concordant = 0 # i experienced the event before j and was given a higher predicted score
    n_tied = 0 # i and j experienced the event at the same time or if the predicted score is the same

    for i in range(len(event_times)):
        for j in range(len(event_times)):
            if i == j:
                continue
            if event_times[i] < event_times[j] and event_observed[i] == 1:
                n += 1
                if predicted_scores[i] > predicted_scores[j]:
                    n_concordant += 1
                elif predicted_scores[i] == predicted_scores[j]:
                    n_tied += 1

    if n == 0:
        return np.nan
    return (n_concordant + 0.5 * n_tied) / n

