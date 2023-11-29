import argparse
import json

from .importer import DatasetsImporter
from .loader import DatasetLoader


def _list():
    result = DatasetLoader._allNamesImported()
    print(f"Found {len(result)} imported datasets.")
    if len(result) > 0:
        print(json.dumps(result, indent=2))


def _import(
    inputFilePath,
    sampleSizePercent=None,
    sampleRandomState=None,
    computePca=False,
    computeTsne=False,
    computeUmap=False,
):
    DatasetsImporter.importDataset(
        inputFilePath,
        sampleSizePercent=sampleSizePercent,
        sampleRandomState=sampleRandomState,
        computePca=computePca,
        computeTsne=computeTsne,
        computeUmap=computeUmap,
    )


def _delete(name):
    DatasetsImporter.deleteImportedDataset(name)


def main(args):
    if args.command == "list":
        _list()
    elif args.command == "import":
        _import(
            args.file,
            sampleSizePercent=args.sampleSizePercent,
            sampleRandomState=args.sampleRandomState,
            computePca=args.pca,
            computeTsne=args.tsne,
            computeUmap=args.umap,
        )
    elif args.command == "remove":
        _delete(args.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pek.data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List the imported datasets")

    # import <file>
    p_import = subparsers.add_parser("import", help="Import a dataset")
    p_import.add_argument("file", help="dataset file")
    # projections
    p_import.add_argument("-pca", "--pca", help="Tells whether compute PCA projection", action="store_true")
    p_import.add_argument("-tsne", "--tsne", help="Tells whether compute TSNE projection", action="store_true")
    p_import.add_argument("-umap", "--umap", help="Tells whether compute UMAP projection", action="store_true")

    # sampling
    p_import.add_argument(
        "-sampleSizePercent",
        "--sampleSizePercent",
        help="Tells whether extract a sample from the dataset. Only integer from 1 to 100.",
        default=None,
    )
    p_import.add_argument(
        "-sampleRandomState", "--sampleRandomState", help="Random state for sampling. Integer.", default=None
    )

    # delete <name>
    p_remove = subparsers.add_parser("delete", help="delete an imported a dataset")
    p_remove.add_argument("name", help="dataset name")

    main(parser.parse_args())
