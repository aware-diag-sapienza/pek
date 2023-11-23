import numpy as np
import scipy.sparse as sp
from sklearn import metrics as skmetrics
from sklearn.cluster._k_means_common import _inertia_dense, _inertia_sparse
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.validation import _check_sample_weight

from ..utils.clustering import getClusters

"""Clustering validation metrics."""


def calinskiHarabasz(data, labels) -> float:
    """Calinski and Harabasz Score. Better max."""
    result = skmetrics.calinski_harabasz_score(data, labels)
    return float(result)  # convert np.float64 to float


def daviesBouldinIndex(data, labels) -> float:
    """Davies Bouldin Index. Better min."""
    result = skmetrics.davies_bouldin_score(data, labels)
    return float(result)  # convert np.float64 to float


def dunnIndex(data, labels) -> float:
    """Dunn Index. Better max."""
    clusters, centers = getClusters(data, labels)
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
    return float(result)  # convert np.float64 to float


def inertia(data, labels) -> float:
    """Inertia. Sum of squared distance between each sample and its assigned center. Better min."""
    if sp.issparse(data):
        _inertia_fn = _inertia_sparse
    else:
        _inertia_fn = _inertia_dense

    # print(data.shape, data.dtype, data.flags)
    # print(labels.shape, labels.dtype, labels.flags)
    # print("\n\n\n")

    clusters, centers = getClusters(data, labels)
    sample_weight = _check_sample_weight(None, data, dtype=data.dtype)
    n_threads = _openmp_effective_n_threads()
    result = _inertia_fn(data, sample_weight, centers, labels.astype(np.int32), n_threads)
    return float(result)  # convert np.float64 to float


def silhouette(data, labels) -> float:
    """Silhouette score. Better max."""
    result = skmetrics.silhouette_score(data, labels)
    return float(result)  # convert np.float64 to float


def simplifiedSilhouette(data, labels) -> float:
    """Simplified Silhouette Coefficient of all samples. Better max."""
    n = data.shape[0]
    clusters, centers = getClusters(data, labels)
    distances = skmetrics.pairwise.euclidean_distances(data, centers)  # distance of each point to all centroids

    A = distances[np.arange(n), labels]  # distance of each point to its cluster centroid
    distances[np.arange(n), labels] = np.Inf  # set to infinite the distance to own centroid

    B = np.min(
        distances, axis=1
    )  # distance to each point to the second closer centroid (different from its own cluster)
    M = np.maximum(A, B)  # max row wise of A and B
    S = np.mean((B - A) / M)
    return float(S)


ALL_VALIDATION_METRICS_DICT = {
    "calinski_harabasz": calinskiHarabasz,
    "davies_bouldin": daviesBouldinIndex,
    "dunn_index": dunnIndex,
    "inertia": inertia,
    "silhouette": silhouette,
    "simplified_silhouette": simplifiedSilhouette,
}

ALL_VALIDATION_METRICS = sorted(ALL_VALIDATION_METRICS_DICT.keys())


def _checkValidationMetric(metricName):
    """
    Validate a given metric name against the predefined dictionary of valid validation metrics.

    Parameters:
    - metricName (str): The name of the validation metric to be checked.

    Returns:
    - The validation metric function.

    Raises:
    - InvalidParameterError: If the provided metric name is not found in the predefined dictionary,
      indicating that the validation metric does not exist.
    """
    if metricName in ALL_VALIDATION_METRICS_DICT:
        return ALL_VALIDATION_METRICS_DICT[metricName]
    else:
        raise InvalidParameterError(f"The '{metricName}' validation metric does not exist.")


def _toValidationMetricDict(metricNameArr):
    """Generate a dict of validation metrics {name: function} given the arrays of names."""
    if metricNameArr is None:
        return {}

    if metricNameArr == "ALL":
        return ALL_VALIDATION_METRICS_DICT

    result = {}
    for metricName in metricNameArr:
        result[metricName] = _checkValidationMetric(metricName)

    return result


__all__ = [
    "ALL_VALIDATION_METRICS",
    "ALL_VALIDATION_METRICS_DICT",
    "calinskiHarabasz",
    "daviesBouldinIndex",
    "dunnIndex",
    "inertia",
    "silhouette",
    "simplifiedSilhouette",
]
