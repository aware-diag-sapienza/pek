import os
from abc import ABC
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..version import __version__
from .folders import Folders

_dtype_str = h5py.special_dtype(vlen=str)


class _Colors:
    ENDC = "\033[0m"
    GRAY = "\033[90m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    PINK = "\033[95m"


def _computeScaledData(data, dtype=float):
    print(f"\tScaling data ...")
    dataScaled = np.asarray(StandardScaler().fit_transform(data), dtype=dtype, order="C")
    return dataScaled


def _computePCA(dataScaled, dtype=float):
    print(f"\tComputing PCA ...")
    if dataScaled.shape[1] > 2:
        pca_proj = PCA(n_components=2, random_state=0).fit_transform(dataScaled)
        pca_proj = np.asarray(MinMaxScaler().fit_transform(pca_proj), dtype=dtype, order="C")
        return pca_proj
    else:
        return dataScaled


def _computeTSNE(dataScaled, dtype=float):
    print(f"\tComputing TSNE ...")
    if dataScaled.shape[1] > 2:
        tsne_proj = TSNE(n_components=2, random_state=0).fit_transform(dataScaled)
        tsne_proj = np.asarray(MinMaxScaler().fit_transform(tsne_proj), dtype=dtype, order="C")
        return tsne_proj
    else:
        return dataScaled


def _computeUMAP(dataScaled, dtype=float):
    print(f"\tComputing UMAP ...")
    from umap import UMAP

    if dataScaled.shape[1] > 2:
        # random_state = 0 --> no random state for parallelism
        umap_proj = UMAP(random_state=0).fit_transform(dataScaled)
        umap_proj = np.asarray(MinMaxScaler().fit_transform(umap_proj), dtype=dtype, order="C")
        return umap_proj
    else:
        return dataScaled


def _getDatasetFile(name):
    return Folders.importedDatasetsFolder().joinpath(f"{name}.hdf5")


class DatasetsImporter(ABC):
    @staticmethod
    def deleteImportedDataset(name):
        print(f"Deleting {name} ...")
        file = _getDatasetFile(name)
        if file.exists():
            os.remove(file)
        else:
            raise NameError(f"The dataset '{name}' is not an imported dataset.")

    @staticmethod
    def importDataset(
        inputFilePath,
        sampleSizePercent=None,
        sampleRandomState=None,
        computePca=True,
        computeTsne=False,
        computeUmap=False,
        **kwargs,
    ) -> Path:
        """Import a csv dataset."""
        inputFilePath = Path(inputFilePath)
        datasetName = inputFilePath.stem
        outputFilePath = _getDatasetFile(datasetName)
        dtype = float  # important !!!

        if not inputFilePath.exists():
            raise RuntimeError(f"The file {inputFilePath.resolve()} does not exist.")

        if sampleSizePercent is not None:
            if int(sampleSizePercent) <= 0 or int(sampleSizePercent) > 100:
                raise RuntimeError(f"Invalid sample size percent {sampleSizePercent}.")

        if computeUmap:
            try:
                from umap import UMAP
            except ImportError:
                print(f"{_Colors.RED}ERROR: To compute UMAP you need to install the umap-learn package.{_Colors.ENDC}")
                print("See details at: https://pypi.org/project/umap-learn/")
                exit()

        print(f"Importing {inputFilePath.stem} ...")
        # create HDF5 file

        with h5py.File(outputFilePath, "w") as hf:
            info = hf.create_dataset("__info__", data=np.zeros(1), compression="gzip", chunks=True)
            info.attrs["__version__"] = __version__
            if sampleSizePercent is not None:
                info.attrs["sampleSize"] = sampleSizePercent
                info.attrs["sampleRandomState"] = sampleRandomState

            print(f"\tLoading input file ...")
            df = pd.read_csv(inputFilePath)

            # features
            features = np.asarray(list(df.columns), dtype=_dtype_str, order="C")
            hf.create_dataset("features", data=features, compression="gzip", chunks=True)

            # data
            data = np.asarray(df.to_numpy(dtype=dtype), order="C")

            if sampleSizePercent is not None:
                totLen = data.shape[0]
                sampledLen = int(np.ceil(totLen * float(sampleSizePercent) / 100))
                print(f"\tSampling to {sampledLen} entries...")
                data = np.random.default_rng(sampleRandomState).choice(data, sampledLen, replace=False)

            hf.create_dataset("data", data=data, compression="gzip", chunks=True)

            # projections
            dataScaled = None
            if computePca:
                if dataScaled is None:
                    dataScaled = _computeScaledData(data, dtype=dtype)
                pca_proj = _computePCA(dataScaled, dtype=dtype)
                hf.create_dataset("pca_proj", data=pca_proj, compression="gzip", chunks=True)

            if computeTsne:
                if dataScaled is None:
                    dataScaled = _computeScaledData(data, dtype=dtype)
                tsne_proj = _computeTSNE(dataScaled, dtype=dtype)
                hf.create_dataset("tsne_proj", data=tsne_proj, compression="gzip", chunks=True)

            if computeUmap:
                if dataScaled is None:
                    dataScaled = _computeScaledData(data, dtype=dtype)
                umap_proj = _computeUMAP(dataScaled, dtype=dtype)
                hf.create_dataset("umap_proj", data=umap_proj, compression="gzip", chunks=True)

            hf.flush()
            hf.close()

        return outputFilePath
