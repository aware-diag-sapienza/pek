from abc import ABC

from .dataset import Dataset

"""List of all available datasets in form of dictionary: {DatasetName: numberOfClusters, ...}"""
_available = {
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


class DatasetLoader(ABC):
    @staticmethod
    def allNames() -> list:
        """Returns the list of all available dataset names."""
        return sorted(_available.keys())

    @staticmethod
    def load(name) -> Dataset:
        """Loads a built-in dataset given the name."""
        if name not in _available:
            raise ValueError(f"Dataset '{name}' does not exist.")
        return Dataset(name, n_clusters=_available[name])

    @staticmethod
    def loadAll() -> list[Dataset]:
        """Loads all the available datasets."""
        return [DatasetLoader.load(n) for n in DatasetLoader.allNames()]
