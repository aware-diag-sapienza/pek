import pkgutil
from io import StringIO as StringIO

import numpy as np
import pandas as pd


def _to_ndarray(arr, dtype):
    if type(arr) == np.ndarray and arr.dtype == dtype:
        return arr
    else:
        return np.asarray(arr, dtype=dtype)


def floatArr(arr):
    """Converts arr to a float ndarray. If arr is already a float ndarray, returns it."""
    return _to_ndarray(arr, float)


def intArray(arr):
    """Converts arr to an int ndarray. If arr is already an int ndarray, returns it."""
    return _to_ndarray(arr, np.int32)
