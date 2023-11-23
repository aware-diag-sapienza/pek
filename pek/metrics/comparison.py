import numpy as np
import sklearn.metrics as _skmetrics

# from sklearn.metrics import a
from sklearn.utils._param_validation import InvalidParameterError


def ari(labels_a, labels_b):
    """Rand index adjusted for chance.
    See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html"""
    return _skmetrics.adjusted_rand_score(np.asarray(labels_a, dtype=int), np.asarray(labels_b, dtype=int))


def ami(labels_a, labels_b):
    """Adjusted Mutual Information between two clusterings.
    See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html"""
    return _skmetrics.adjusted_mutual_info_score(np.asarray(labels_a, dtype=int), np.asarray(labels_b, dtype=int))


ALL_COMPARISON_METRICS_DICT = {"ari": ari, "ami": ami}
ALL_COMPARISON_METRICS = sorted(ALL_COMPARISON_METRICS_DICT.keys())


def _checkComparisonMetric(metricName):
    """
    Validate a given metric name against the predefined dictionary of valid comparison metrics.

    Parameters:
    - metricName (str): The name of the comparison metric to be checked.

    Returns:
    - The comparison metric function.

    Raises:
    - InvalidParameterError: If the provided metric name is not found in the predefined dictionary,
      indicating that the comparison metric does not exist.
    """
    if metricName in ALL_COMPARISON_METRICS_DICT:
        return ALL_COMPARISON_METRICS_DICT[metricName]
    else:
        raise InvalidParameterError(f"The '{metricName}' comparison metric does not exist.")


def _toComparisonMetricDict(metricNameArr):
    """Generate a dict of comparison metrics {name: function} given the arrays of names."""
    if metricNameArr is None:
        return {}

    if metricNameArr == "ALL":
        return ALL_COMPARISON_METRICS_DICT

    result = {}
    for metricName in metricNameArr:
        result[metricName] = _checkComparisonMetric(metricName)

    return result


__all__ = ["ALL_COMPARISON_METRICS", "ALL_COMPARISON_METRICS_DICT", "ari", "ami"]
