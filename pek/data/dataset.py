import pkgutil
from abc import ABC
from io import BytesIO, StringIO

import numpy as np
from sklearn.utils import Bunch

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

    def __init__(self, name):
        self._name = name
        self._header = None
        self._n_clusters = None

        self._data = None
        self._data_scaled = None

        self._pca = None
        self._tsne = None
        self._umap = None

    def toDict(self, insertData=True, insertProjections=True):
        d = Bunch(
            name=self.name,
            header=self.header,
            n_clusters=self.n_clusters,
            # data=self.data,
            # data_scaled=self.data_scaled,
            # projections=Bunch(pca=self.pca, tsne=self.tsne, umap=self.umap),
        )

        if insertData:
            d["data"] = self.data
            d["data_scaled"] = self.data_scaled

        if insertProjections:
            d["projections"] = Bunch(pca=self.pca, tsne=self.tsne, umap=self.umap)

        return d

    @property
    def name(self):
        return self._name

    @property
    def header(self):
        if self._header is None:
            self._header = _loadPackageFile_npy(f"_npy/{self.name}.header.npy")
        return self._header

    @property
    def n_clusters(self):
        if self._n_clusters is None:
            self._n_clusters = _default_num_clusters[self.name]
        return self._n_clusters

    @property
    def data(self):
        if self._data is None:
            self._data = _loadPackageFile_npy(f"_npy/{self.name}.npy")
        return self._data

    @property
    def data_scaled(self):
        if self._data_scaled is None:
            self._data_scaled = _loadPackageFile_npy(f"_npy/{self.name}.scaled.npy")
        return self._data_scaled

    @property
    def pca(self):
        if self._pca is None:
            self._pca = _loadPackageFile_npy(f"_npy/{self.name}.pca.npy")
        return self._pca

    @property
    def tsne(self):
        if self._tsne is None:
            self._tsne = _loadPackageFile_npy(f"_npy/{self.name}.tsne.npy")
        return self._tsne

    @property
    def umap(self):
        if self._umap is None:
            self._umap = _loadPackageFile_npy(f"_npy/{self.name}.umap.npy")
        return self._umap

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
        """Return a built-in dataset given the name."""
        _checkName(name)
        return _BuiltInDataset(name)
