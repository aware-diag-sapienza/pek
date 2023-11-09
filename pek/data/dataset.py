import pkgutil
from abc import ABC
from io import BytesIO, StringIO

import numpy as np

_names = [
    "A1",
    "A2",
    "A3",
    "BalanceScale",
    "ContraceptiveMethodChoice",
    "Diabetes",
    "Glass",
    "HeartStatlog",
    "Ionosphere",
    "Iris",
    "LiverDisorder",
    "S1",
    "S2",
    "S3",
    "S4",
    "Segmentation",
    "Sonar",
    "SpectfHeart",
    "Unbalanced",
    "Vehicles",
    "Wine",
]

_default_num_clusters = {
    "A1": 20,
    "A2": 35,
    "A3": 50,
    "BalanceScale": 3,
    "ContraceptiveMethodChoice": 3,
    "Diabetes": 2,
    "Glass": 6,
    "HeartStatlog": 2,
    "Ionosphere": 2,
    "Iris": 3,
    "LiverDisorder": 2,
    "S1": 15,
    "S2": 15,
    "S3": 15,
    "S4": 15,
    "Segmentation": 7,
    "Sonar": 2,
    "SpectfHeart": 2,
    "Unbalanced": 8,
    "Vehicles": 4,
    "Wine": 3,
}


def _checkName(name):
    """Checks if the name of the dataset exists."""
    if name not in _names:
        raise ValueError(f"Dataset '{name}' does not exist.")


'''def _loadDataframe(datasetName) -> pd.DataFrame:
    """Loads the dataset in form of pandas dataframe. Read the csv file."""
    _checkName(datasetName)
    file = f"_csv/{datasetName}.csv"
    csvContent = str(pkgutil.get_data(__name__, file).decode())
    df = pd.DataFrame(StringIO(csvContent))
    return df
'''


def _loadPackageFile_npy(filePath) -> np.ndarray:
    """Loads a npy file located inside the package."""
    return np.load(BytesIO(pkgutil.get_data(__name__, filePath)))


class _BuiltInDataset:
    """A class representing a built-in dataset."""

    def __init__(self, name, data, header=None, data_scaled=None, n_clusters=None, pca=None, tsne=None, umap=None):
        self.name = name
        self.data = data

        self.header = header
        self.data_scaled = data_scaled if data_scaled is not None else self.data
        self.n_clusters = n_clusters
        self.pca = pca
        self.tsne = tsne
        self.umap = umap

    def __str__(self):
        return f"{self.__class__.__name__}<{self.name}> shape={self.data.shape}"


class BuiltInDatasetLoader(ABC):
    @staticmethod
    def allNames() -> list:
        """Returns the list of all available datasets."""
        return _names

    @staticmethod
    def all() -> list[_BuiltInDataset]:
        """Returns the list of all available datasets objects."""
        return [BuiltInDatasetLoader.load(n) for n in _names]

    @staticmethod
    def load(name) -> _BuiltInDataset:
        """Return a dataset given the name.
        The dataset is dictionary: {'name': str, 'n_clusters': int, 'data': ndarray}"""
        _checkName(name)
        d = _BuiltInDataset(
            name,
            _loadPackageFile_npy(f"_npy/{name}.npy"),
            header=_loadPackageFile_npy(f"_npy/{name}.header.npy"),
            data_scaled=_loadPackageFile_npy(f"_npy/{name}.scaled.npy"),
            n_clusters=_default_num_clusters[name],
            pca=_loadPackageFile_npy(f"_npy/{name}.pca.npy"),
            tsne=_loadPackageFile_npy(f"_npy/{name}.tsne.npy"),
            umap=_loadPackageFile_npy(f"_npy/{name}.umap.npy"),
        )
        return d
