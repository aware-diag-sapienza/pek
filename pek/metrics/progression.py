import numpy as np
from sklearn.utils._param_validation import InvalidParameterError


def _entries_stability(labelsHistoryArr, window=None):
    """Stability of labels for each data entry,
    considering the array of labels of the iterations and the window (last elements). If none, consider all labels.."""

    if (window is not None) and (window < 2):
        raise InvalidParameterError(f"Parameter window must be >= 2. Got {window} instead.")

    if window is None:
        hist = labelsHistoryArr
    else:
        hist = labelsHistoryArr[-window:]

    stability = np.full_like(labelsHistoryArr[0], 0, dtype=float)
    h = len(labelsHistoryArr)
    w = [np.log(2 + i) for i in range(h - 1)]  # log weights
    for i in range(h - 1):
        stability += ((labelsHistoryArr[h - 1] == labelsHistoryArr[i]).astype(float) * w[i]) / sum(w)
    return stability


def _global_stability(labelsHistory, window=None):
    """Mean stability of labels for all the data entries."""
    est = _entries_stability(labelsHistory, window)
    return float(np.mean(est))


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


def entries_stability_2(labelsHistoryArr):
    return _entries_stability(labelsHistoryArr, 2)


def entries_stability_3(labelsHistoryArr):
    return _entries_stability(labelsHistoryArr, 3)


def entries_stability_4(labelsHistoryArr):
    return _entries_stability(labelsHistoryArr, 3)


def entries_stability_5(labelsHistoryArr):
    return _entries_stability(labelsHistoryArr, 5)


def entries_stability_10(labelsHistoryArr):
    return _entries_stability(labelsHistoryArr, 10)


def entries_stability_all(labelsHistoryArr):
    return _entries_stability(labelsHistoryArr, None)


def global_stability_2(labelsHistoryArr):
    return _global_stability(labelsHistoryArr, 2)


def global_stability_3(labelsHistoryArr):
    return _global_stability(labelsHistoryArr, 3)


def global_stability_4(labelsHistoryArr):
    return _global_stability(labelsHistoryArr, 4)


def global_stability_5(labelsHistoryArr):
    return _global_stability(labelsHistoryArr, 5)


def global_stability_10(labelsHistoryArr):
    return _global_stability(labelsHistoryArr, 10)


def global_stability_all(labelsHistoryArr):
    return _global_stability(labelsHistoryArr, None)


ALL_PROGRESSION_METRICS_DICT = {
    "entries_stability_2": entries_stability_2,
    "entries_stability_3": entries_stability_3,
    "entries_stability_4": entries_stability_4,
    "entries_stability_5": entries_stability_5,
    "entries_stability_10": entries_stability_10,
    "entries_stability_all": entries_stability_all,
    "global_stability_2": global_stability_2,
    "global_stability_3": global_stability_3,
    "global_stability_4": global_stability_4,
    "global_stability_5": global_stability_5,
    "global_stability_10": global_stability_10,
    "global_stability_all": global_stability_all,
}
ALL_PROGRESSION_METRICS = sorted(ALL_PROGRESSION_METRICS_DICT.keys())


def _checkProgressionMetric(metricName):
    """
    Validate a given metric name against the predefined dictionary of valid progression metrics.

    Parameters:
    - metricName (str): The name of the progression metric to be checked.

    Returns:
    - The comparison metric function.

    Raises:
    - InvalidParameterError: If the provided metric name is not found in the predefined dictionary,
      indicating that the progression metric does not exist.
    """
    if metricName in ALL_PROGRESSION_METRICS_DICT:
        return ALL_PROGRESSION_METRICS_DICT[metricName]
    else:
        raise InvalidParameterError(f"The '{metricName}' comparison metric does not exist.")


def _toProgressionMetricDict(metricNameArr):
    """Generate a dict of progression metrics {name: function} given the arrays of names."""
    if metricNameArr is None:
        return {}

    if metricNameArr == "ALL":
        return ALL_PROGRESSION_METRICS_DICT

    result = {}
    for metricName in metricNameArr:
        result[metricName] = _checkProgressionMetric(metricName)

    return result


__all__ = [
    "ALL_PROGRESSION_METRICS",
    "ALL_PROGRESSION_METRICS_DICT",
    "entries_stability_2",
    "entries_stability_3",
    "entries_stability_4",
    "entries_stability_5",
    "entries_stability_10",
    "entries_stability_all",
    "global_stability_2",
    "global_stability_3",
    "global_stability_4",
    "global_stability_5",
    "global_stability_10",
    "global_stability_all",
]
