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
    folder = Folders.importedDatasetsFolder()
    filePath = folder.joinpath(f"{name}.hdf5")
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
            "Unbalanced",
            "Vehicles",
            "Wine",
        ]
        return ls

    @staticmethod
    def _allNamesImported():
        """List of all imported datasets."""
        folder = Folders.importedDatasetsFolder(createIfNotExist=False)
        print(folder.resolve())
        if folder.exists():
            return sorted([file.stem for file in folder.glob(f"*.hdf5")])
        else:
            return []

    @staticmethod
    def allNames() -> list:
        """Returns the list of all available dataset names."""
        ls = set()
        for d in DatasetLoader._allNamesInPackage():
            ls.add(d)
        for d in DatasetLoader._allNamesImported():
            ls.add(d)

        completeList = sorted(ls)
        # completeList = sorted(list(set([DatasetLoader._allNamesInPackage() + DatasetLoader._allNamesImported()])))
        # completeList = sorted(list([DatasetLoader._allNamesInPackage() + DatasetLoader._allNamesImported()}))
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
    def loadAll() -> list[Dataset]:
        """Loads all the available datasets."""
        return [DatasetLoader.load(n) for n in DatasetLoader.allNames()]
