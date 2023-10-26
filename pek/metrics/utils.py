import numpy as np


def _getClusters(data, labels):
    """
    Returns a tuple (clusters, centers). If we have k clusters:
    - clusters: an array [c_1, ..., c_k] where c_i is the cluster i as ndarray (subset of data).
    - centers: an array [c_1, ..., c_k] where ci is the center of the cluster i.
    """
    unique_labels = np.unique(labels)
    clusters_idx = [np.where(labels == l) for l in unique_labels]
    clusters = [data[i] for i in clusters_idx]
    centers = np.array([np.mean(c, axis=0) for c in clusters], dtype=float)
    return clusters, centers
