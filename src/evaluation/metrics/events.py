import numpy as np
from src.evaluation.metrics.utils import remove_missing



def c_index(config, labels, preds, sample_weights=None):

    labels, preds = remove_missing(config,labels,preds)

    events, days = labels[:, 0], labels[:, 1]  # unpack the events and days from the labels
    # Ensure preds is a 1D array
    # preds = np.asarray(preds).flatten()
    # # Ensure events and days are 1D arrays
    # events = np.asarray(events).flatten()
    # days = np.asarray(days).flatten()

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

    n = 0
    n_concordant = 0
    n_tied = 0

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

