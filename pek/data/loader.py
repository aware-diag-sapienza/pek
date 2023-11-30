import pkgutil
from abc import ABC
from io import BytesIO

import h5py

from .dataset import Dataset
from .folders import Folders


def _loadInPackageDataset(name):
    filePath = Folders.packageDataFolder() + f"{name}.hdf5"
    input = BytesIO(pkgutil.get_data(__name__, filePath))
    file = h5py.File(input, "r")
    return Dataset(name, file)


def _loadImportedDataset(name):
    folder_local = Folders.importedDatasetsFolder(createIfNotExist=False)
    filePath = folder_local.joinpath(f"{name}.hdf5")
    if filePath.exists():
        file = h5py.File(filePath, "r")
        return Dataset(name, file)

    folder_home = Folders.importedDatasetsFolderHome(createIfNotExist=False)
    filePath = folder_home.joinpath(f"{name}.hdf5")
    if filePath.exists():
        file = h5py.File(filePath, "r")
        return Dataset(name, file)


class DatasetLoader(ABC):
    @staticmethod
    def _allNamesInPackage():
        """List of all available datasets in the package."""
        ls = [
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
            "SpotifySong",
            # "SpotifySongFull",
            "Unbalanced",
            "Vehicles",
            "Wine",
        ]
        return ls

    @staticmethod
    def _allNamesImported():
        """List of all imported datasets."""
        result = set()

        folder_local = Folders.importedDatasetsFolder(createIfNotExist=False)
        if folder_local.exists():
            for file in folder_local.glob(f"*.hdf5"):
                result.add(file.stem)

        folder_home = Folders.importedDatasetsFolderHome(createIfNotExist=False)
        if folder_home.exists():
            for file in folder_home.glob(f"*.hdf5"):
                result.add(file.stem)

        return sorted(list(result))

    @staticmethod
    def allNames() -> list:
        """Returns the list of all available dataset names."""
        ls = set()
        for d in DatasetLoader._allNamesInPackage():
            ls.add(d)
        for d in DatasetLoader._allNamesImported():
            ls.add(d)

        completeList = sorted(ls)
        return completeList

    @staticmethod
    def load(name) -> Dataset:
        """Loads a dataset given the name."""
        if name not in set(DatasetLoader.allNames()):
            raise ValueError(f"Dataset '{name}' does not exist.")

        if name in DatasetLoader._allNamesInPackage():
            return _loadInPackageDataset(name)
        elif name in DatasetLoader._allNamesImported():
            return _loadImportedDataset(name)
        else:
            raise ValueError(f"Dataset '{name}' does not exist.")

    @staticmethod
    def loadAll() -> list:
        """Loads all the available datasets."""
        return [DatasetLoader.load(n) for n in DatasetLoader.allNames()]
