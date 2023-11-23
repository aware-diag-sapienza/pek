import json
import pkgutil
from io import BytesIO

import numpy as np
from sklearn.utils import Bunch

from ..utils.encoding import NumpyEncoder


def _loadPackageFile_npy(filePath) -> np.ndarray:
    """Loads a npy file located inside the package."""
    return np.load(BytesIO(pkgutil.get_data(__name__, filePath)))


class Dataset:
    """A class representing a built-in dataset."""

    def __init__(self, name, n_clusters=None):
        self._name = name
        self._header = None
        self._n_clusters = n_clusters

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
        )

        if insertData:
            d["data"] = self.data
            d["data_scaled"] = self.data_scaled

        if insertProjections:
            d["projections"] = Bunch(pca=self.pca, tsne=self.tsne, umap=self.umap)

        return d

    def toJson(self, insertData=True, insertProjections=True, indent=None):
        return json.dumps(
            self.toDict(insertData=insertData, insertProjections=insertProjections), cls=NumpyEncoder, indent=indent
        )

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
