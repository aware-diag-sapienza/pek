import argparse
import json

from .importer import DatasetsImporter
from .loader import DatasetLoader


def _list():
    result = DatasetLoader._allNamesImported()
    print(f"Found {len(result)} imported datasets.")
    if len(result) > 0:
        print(json.dumps(result, indent=2))


def _import(inputFilePath, computePca=False, computeTsne=False, computeUmap=False):
    DatasetsImporter.importDataset(
        inputFilePath, computePca=computePca, computeTsne=computeTsne, computeUmap=computeUmap
    )


def _delete(name):
    DatasetsImporter.deleteImportedDataset(name)


def main(args):
    if args.command == "list":
        _list()
    elif args.command == "import":
        _import(args.file, computePca=args.pca, computeTsne=args.tsne, computeUmap=args.umap)
    elif args.command == "remove":
        _delete(args.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pek.data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="list the imported datasets")

    # import <file>
    p_import = subparsers.add_parser("import", help="import a dataset")
    p_import.add_argument("file", help="dataset file")
    p_import.add_argument("-pca", "--pca", help="tells whether compute PCA projection", action="store_true")
    p_import.add_argument("-tsne", "--tsne", help="tells whether compute TSNE projection", action="store_true")
    p_import.add_argument("-umap", "--umap", help="tells whether compute UMAP projection", action="store_true")

    # delete <name>
    p_remove = subparsers.add_parser("delete", help="delete an imported a dataset")
    p_remove.add_argument("name", help="dataset name")

    main(parser.parse_args())
