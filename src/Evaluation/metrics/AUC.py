import logging
import math
import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_auc(outputs, targets):
    if np.unique(targets).size > 1:
        auc = roc_auc_score(targets, outputs)
    else:
        auc = None
        logging.warning(f'Cannot compute ROC AUC score because targets contain only one class')
    return auc