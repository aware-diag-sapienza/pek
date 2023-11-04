import pkgutil
from abc import ABC
from io import StringIO as StringIO

import numpy as np
from sklearn.utils import Bunch

# the file names is: datasetName_numClusters.csv
_files = [
    "A1_20.csv",
    "A2_35.csv",
    "A3_50.csv",
    "BalanceScale_3.csv",
    "ContraceptiveMethodChoice_3.csv",
    "Diabetes_2.csv",
    "Glass_6.csv",
    "HeartStatlog_2.csv",
    "Ionosphere_2.csv",
    "Iris_3.csv",
    "LiverDisorder_2.csv",
    "S1_15.csv",
    "S2_15.csv",
    "S3_15.csv",
    "S4_15.csv",
    "Segmentation_7.csv",
    "Sonar_2.csv",
    "SpectfHeart_2.csv",
    "Unbalanced_8.csv",
    "Vehicles_4.csv",
    "Wine_3.csv",
]


def _checkName(name):
    """Checks if the name of the dataset exists."""
    allNames = Dataset.allNames()
    if name not in allNames:
        raise ValueError(f"Dataset '{name}' does not exist.")


def _loadData(datasetName) -> np.ndarray:
    """Loads the dataset in form of numpy array."""
    _checkName(datasetName)
    for f in _files:
        if f.startswith(datasetName + "_"):
            csvContent = str(pkgutil.get_data(__name__, f"_csvData/{f}").decode())
            X = np.loadtxt(StringIO(csvContent), skiprows=1, delimiter=",").astype(float)
            return X


class _Dataset(Bunch):
    def __init__(self, name, n_clusters, data):
        super().__init__(name=name, n_clusters=n_clusters, data=data)


class Dataset(ABC):
    """Utility class for loading built-in datasets."""

    @staticmethod
    def allNames() -> list:
        """Returns the list of all available datasets."""
        result = []
        for s in _files:
            name = s.split("_")[0]
            result.append(name)
        return result

    @staticmethod
    def all() -> list:
        """Returns the list of all available datasets."""
        result = []
        for s in _files:
            name = s.split("_")[0]
            result.append(Dataset.get(name))
        return result

    @staticmethod
    def get(name) -> _Dataset:
        """Return a dataset given the name.
        The dataset is dictionary: {'name': str, 'n_clusters': int, 'data': ndarray}"""
        _checkName(name)
        for s in _files:
            if s.split("_")[0] == name:
                n_clusters = int(s.split("_")[1].replace(".csv", ""))
                data = _loadData(name)
                return _Dataset(name, n_clusters, data)
