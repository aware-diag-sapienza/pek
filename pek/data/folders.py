from abc import ABC
from pathlib import Path


class Folders(ABC):
    @staticmethod
    def packageDataFolder() -> str:
        return "_hdf5/"

    @staticmethod
    def importedDatasetsFolder(createIfNotExist=True) -> Path:
        """Imported datasets folder in current directory."""
        folder = Path("pek_data").joinpath("datasets")
        if createIfNotExist:
            folder.mkdir(exist_ok=True, parents=True)
        return folder

    @staticmethod
    def importedDatasetsFolderHome(createIfNotExist=True) -> Path:
        """Imported datasets folder in the home directory."""
        folder = Path.home().joinpath("pek_data").joinpath("datasets")
        if createIfNotExist:
            folder.mkdir(exist_ok=True, parents=True)
        return folder
