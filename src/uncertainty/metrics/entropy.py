import numpy as np


def binary_entropy(p):
    
    eps = 1e-8
    return -p * np.log2(p + eps) - (1 - p) * np.log2(1 - p + eps)

