import numpy as np
import scipy.sparse as sp
from sklearn import metrics as skmetrics
from sklearn.cluster._k_means_common import _inertia_dense, _inertia_sparse
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.validation import _check_sample_weight


def _getClusters(data, labels):
    """
    Returns a tuple (clusters, centers). If we have k clusters:
    - clusters: an array [c1, ..., ck] where ci is the cluster i as ndarray (subset of data).
    - centers: an array [c1, ..., ck] where ci is the center of the cluster i
    """
    unique_labels = np.unique(labels)
    clusters_idx = [np.where(labels == l) for l in unique_labels]
    clusters = [data[i] for i in clusters_idx]
    centers = np.array([np.mean(c, axis=0) for c in clusters], dtype=float)
    return clusters, centers


def calinskiHarabasz(data, labels) -> float:
    """Calinski and Harabasz Score. Better max."""
    return skmetrics.calinski_harabasz_score(data, labels)


def daviesBouldinIndex(data, labels) -> float:
    """Davies Bouldin Index. Better min."""
    return skmetrics.davies_bouldin_score(data, labels)


def dunnIndex(data, labels) -> float:
    """Dunn Index. Better max."""
    clusters, centers = _getClusters(data, labels)
    centers_pairwise_distances = skmetrics.pairwise.euclidean_distances(centers)

    max_cluster_diameter = 0
    for k in range(len(clusters)):
        cluster = clusters[k]
        center = centers[k]
        distances = skmetrics.pairwise.euclidean_distances(cluster, [center])
        max_cluster_diameter = max(np.mean(distances), max_cluster_diameter)

    idx = np.triu_indices(centers_pairwise_distances.shape[0], 1)
    min_centers_distance = np.min(centers_pairwise_distances[idx])
    result = min_centers_distance / max_cluster_diameter
    return result


def inertia(data, labels) -> float:
    """Inertia. Sum of squared distance between each sample and its assigned center. Better min."""
    if sp.issparse(data):
        _inertia_fn = _inertia_sparse
    else:
        _inertia_fn = _inertia_dense

    clusters, centers = _getClusters(data, labels)
    sample_weight = _check_sample_weight(None, data, dtype=data.dtype)
    n_threads = _openmp_effective_n_threads()
    return _inertia_fn(data, sample_weight, centers, labels.astype(np.int32), n_threads)


def silhouette(data, labels) -> float:
    """Silhouette score. Better max."""

    return skmetrics.silhouette_score(data, labels)


def simplifiedSilhouette(data, labels) -> float:
    """Simplified Silhouette Coefficient of all samples. Better max."""
    n = data.shape[0]
    clusters, centers = _getClusters(data, labels)
    distances = skmetrics.pairwise.euclidean_distances(clusters, centers)  # distance of each point to all centroids

    A = distances[np.arange(n), labels]  # distance of each point to its cluster centroid
    distances[np.arange(n), labels] = np.Inf  # set to infinte the distance to own centroid

    B = np.min(
        distances, axis=1
    )  # distance to each point to the second closer centroid (different from its own cluster)
    M = np.maximum(A, B)  # max row wise of A and B
    S = np.mean((B - A) / M)
    return float(S)


ALL_FN = {
    "calinski_harabasz": calinskiHarabasz,
    "davies_bouldin": daviesBouldinIndex,
    "dunn_index": dunnIndex,
    "inertia": inertia,
    "silhouette": silhouette,
    "simplified_silhouette": simplifiedSilhouette,
}


__all__ = [
    "ALL_FN",
    "calinskiHarabasz",
    "daviesBouldinIndex",
    "dunnIndex",
    "inertia",
    "silhouette",
    "simplifiedSilhouette",
]
