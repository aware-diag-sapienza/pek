from sklearn.utils._param_validation import InvalidParameterError

"""def checkETS(element):
    if element is None:
        return []
    elif isinstance(element, list):
        if all(isinstance(item, AbstractEarlyTerminator) for item in element):
            return element
    raise InvalidParameterError(
        f"The 'ets' parameter must be a list of instances of {AbstractEarlyTerminator.__class__}. Or None"
    )"""


"""def checkValidationMetrics(param):
    if param is None:
        return {}
    for metricName in param:
        if metricName not in ALL_VALIDATION_METRICS:
            raise InvalidParameterError(f"The '{metricName}' validation metric does not exist.")
    return {metricName: ALL_VALIDATION_METRICS[metricName] for metricName in param}


def checkProgressionMetrics(param):
    if param is None:
        return {}
    for metricName in param:
        if metricName not in ALL_PROGRESSION_METRICS:
            raise InvalidParameterError(f"The '{metricName}' progression metric does not exist.")
    return {metricName: ALL_PROGRESSION_METRICS[metricName] for metricName in param}


def checkComparisonMetrics(param):
    if param is None:
        return {}
    for metricName in param:
        if metricName not in ALL_COMPARISON_METRICS:
            raise InvalidParameterError(f"The '{metricName}' comparison metric does not exist.")
    return {metricName: ALL_COMPARISON_METRICS[metricName] for metricName in param}"""


def checkInstance(param, test_class, param_name="parameter", allowsNone=False):
    """
    Check if the parameter is an instance of a specific class.

    Parameters:
    - param: The parameter to be checked.
    - test_class: The class type to check against.
    - param_name (optional): The name of the parameter (for error message).

    Returns:
    - param: The parameter if it is an instance of the specified class.

    Raises:
    - InvalidParameterError: If the parameter is not an instance of the specified class.

    Example:
    >>> checkInstance(42, int, 'my_param')
    # Returns 42.

    >>> checkInstance('invalid', int, 'my_param')
    # Raises InvalidParameterError with the message: 'my_param must be an instance of <class 'int'>.'
    """

    if param is None and allowsNone:
        return param

    if not isinstance(param, test_class):
        raise InvalidParameterError(f"{param_name} must be an instance of {test_class}.")
    return param


"""def getAllParams(lc, exclude=None):
    Returns all the parameters passed to a function.
    The lc parameters must be locals().
    checkInstance(lc, dict, param_name="lc", allowsNone=False)
    checkInstance(exclude, list, param_name="exclude", allowsNone=True)

    filterout = set(["self", "__class__"])
    if exclude is not None:
        filterout = filterout.union(set(exclude))

    result = []

    for key in sorted(lc.keys()):
        if key in filterout:
            continue
        result.append(Bunch(key=key, value=lc[key]))

    return result


def getParamsHash(params):
    Returns the hash of all parameters passed.
    checkInstance(params, list, param_name="params", allowsNone=False)
    for i, p in enumerate(params):
        checkInstance(params, Bunch, param_name=f"params#{i}", allowsNone=False)

    ls = []
    for p in params:
        hv = str(p.value)
        if isinstance(p.value, CacheHashable):
            hv = p.value.__cache_hash__()

        s = str(p.key) + "::" + str(hv)
        ls.append(s)

    ls = "&&".join(ls)
    return hashlib.sha1(ls.encode()).hexdigest()


def getAllParamsHash(lc, exclude=None):
    params = getAllParams(lc, exclude)
    return getParamsHash(params)
"""
