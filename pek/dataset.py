import pkgutil as pkgutil
from abc import ABC
from io import StringIO as StringIO

import numpy as np

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


class Dataset(ABC):
    """Utility class for loading built-in datasets."""

    @staticmethod
    def _checkName(datasetName):
        allNames = Dataset.allNames()
        if datasetName not in allNames:
            raise NameError(f"Dataset name '{datasetName}' is invalid.")

    @staticmethod
    def allNames() -> list:
        """Returns the list of available datasets."""
        return [n.split("_")[0] for n in _fileNames]

    @staticmethod
    def load(datasetName) -> np.ndarray:
        """Loads the dataset in form of numpy array."""
        Dataset._checkName(datasetName)
        for f in _fileNames:
            if f.startswith(datasetName + "_"):
                csvContent = str(pkgutil.get_data(__name__, f"csv/{f}").decode())
                X = np.loadtxt(StringIO(csvContent), skiprows=1, delimiter=",").astype(float)
                return X

    @staticmethod
    def getNumClusters(datasetName):
        """Returns the number of clusters of the dataset."""
        Dataset._checkName(datasetName)
        for f in _fileNames:
            numClusters = int(f.split("_")[1].replace(".csv", ""))
            return numClusters
