import numpy as np


def best_labels_dtype(n_clusters):
    """Best dtype for the number of distinct label existing"""
    if n_clusters <= 255:
        return np.uint8
    elif n_clusters <= 65535:
        return np.uint16
    else:
        return np.uint32
