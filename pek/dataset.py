import pkgutil as pkgutil
from abc import ABC
from io import StringIO as StringIO

import numpy as np
from sklearn.utils import Bunch

# the file names is: datasetName_numClusters.csv
_fileNames = [
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


def _checkName(datasetName):
    allNames = [d.name for d in Dataset.all()]
    if datasetName not in allNames:
        raise NameError(f"Dataset name '{datasetName}' is invalid.")


def _loadData(datasetName) -> np.ndarray:
    """Loads the dataset in form of numpy array."""
    _checkName(datasetName)
    for f in _fileNames:
        if f.startswith(datasetName + "_"):
            csvContent = str(pkgutil.get_data(__name__, f"csv/{f}").decode())
            X = np.loadtxt(StringIO(csvContent), skiprows=1, delimiter=",").astype(float)
            return X


class Dataset:
    """Utility class for loading built-in datasets."""

    def __init__(self, name):
        self.name = name
        self.n_clusters = list(filter(lambda d: d.name == name, Dataset.all()))[0].n_clusters
        self.data = _loadData(name)

    @staticmethod
    def all() -> list:
        """Returns the list of available datasets, reporting the name and the number of clusters.
        [{'name': 'A1', 'n_clusters': 20}, ...]"""
        result = []
        for s in _fileNames:
            name = s.split("_")[0]
            n_clusters = int(s.split("_")[1].replace(".csv", ""))
            result.append(Bunch(name=name, n_clusters=n_clusters))
        return result
