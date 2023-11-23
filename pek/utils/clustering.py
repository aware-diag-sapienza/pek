import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def getClusters(data, labels):
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


"""def best_labels_dtype(n_clusters):
    #Best dtype for the number of distinct label existing
    if n_clusters <= 255:
        return np.uint8
    elif n_clusters <= 65535:
        return np.uint16
    else:
        return np.uint32"""


def adjustLabels(currLabels, currCentroids, prevCentroids):
    """Adjust labels in order to be robust again permutation of labels with the same clustering.
    Looks to the previous centroids to maintain consistence."""
    dist = euclidean_distances(currCentroids, prevCentroids)

    mapping = [None for _ in range(dist.shape[0])]
    while mapping.count(None) != 0:
        i, j = np.unravel_index(dist.argmin(), dist.shape)  # index of min value of distance
        dist[i] = np.inf  # remove row i from matrix (set distance to infinite)
        dist[:, j] = np.inf  # remove row j from matrix (set distance to infinite)
        mapping[i] = j  # currCentroids[i] is mapped to prevCentroids[j] --> label i is mapped to j

    adjustedLabels = currLabels.copy()
    for i, j in enumerate(mapping):
        # label i is mapped to j
        adjustedLabels[currLabels == i] = j

    return adjustedLabels
