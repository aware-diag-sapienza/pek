from pek import ProgressiveEnsembleElbow, ProgressiveEnsembleKMeans
from pek.data import DatasetLoader


def main():
    dataset = DatasetLoader.load("SpotifySong_5P")

    # ENSEMBLE

    ensemble = ProgressiveEnsembleKMeans(
        dataset.data,
        n_clusters=5,
        n_runs=4,
        random_state=0,
        labelsValidationMetrics="ALL",
        labelsComparisonMetrics="ALL",
        labelsProgressionMetrics="ALL",
        partitionsValidationMetrics="ALL",
        partitionsComparisonMetrics="ALL",
        partitionsProgressionMetrics="ALL",
        ets=["slow-kill", "fast-notify"],
        adjustCentroids=False,
        adjustLabels=False,
    )
    while ensemble.hasNextIteration():
        r = ensemble.executeNextIteration()
        print(r.info, r.earlyTermination)

    # ELBOW
    exit()
    elbow = ProgressiveEnsembleElbow(
        dataset.data,
        n_clusters_arr=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        n_runs=16,
        random_state=0,
        labelsValidationMetrics="ALL",
        et="fast-kill",
    )
    while elbow.hasNextIteration():
        r = elbow.executeNextIteration()
        print(r.info)


def test_import():
    from pathlib import Path

    from pek.data.importer import DatasetsImporter

    for f in Path("csv").glob("*.csv"):
        DatasetsImporter.importDataset(f, computePca=True, computeTsne=True, computeUmap=True)


if __name__ == "__main__":
    main()
    # test_import()
