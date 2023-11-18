from sklearn.utils._param_validation import InvalidParameterError

from ..metrics.validation import ALL_VALIDATION_METRICS
from ..termination import AbstractEarlyTerminator


def checkETS(element):
    if element is None:
        return []
    elif isinstance(element, list):
        if all(isinstance(item, AbstractEarlyTerminator) for item in element):
            return element
    raise InvalidParameterError(
        f"The 'ets' parameter must be a list of instances of {AbstractEarlyTerminator.__class__}. Or None"
    )


def checkValidationMetrics(param):
    if param is None:
        return {}
    for metricName in param:
        if metricName not in ALL_VALIDATION_METRICS:
            raise InvalidParameterError(f"The '{metricName}' validation metric name is not a does not exist.")
    return {metricName: ALL_VALIDATION_METRICS[metricName] for metricName in param}
