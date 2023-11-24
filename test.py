from pek import ProgressiveEnsembleElbow, ProgressiveEnsembleKMeans
from pek.data import DatasetLoader


def main():
    dataset = DatasetLoader.load("LiverDisorder")

    # ENSEMBLE

    ensemble = ProgressiveEnsembleKMeans(
        dataset.data,
        n_clusters=6,
        n_runs=16,
        random_state=0,
        labelsValidationMetrics="ALL",
        labelsComparisonMetrics="ALL",
        labelsProgressionMetrics="ALL",
        partitionsValidationMetrics="ALL",
        partitionsComparisonMetrics="ALL",
        partitionsProgressionMetrics="ALL",
        ets=["slow-notify", "fast-notify"],
    )
    while ensemble.hasNextIteration():
        r = ensemble.executeNextIteration()
        print(r.info, r.earlyTermination)

    # ELBOW

    elbow = ProgressiveEnsembleElbow(
        dataset.data,
        n_clusters_arr=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        n_runs=16,
        random_state=0,
        validationMetrics="ALL",
        et="fast-kill",
    )
    while elbow.hasNextIteration():
        r = elbow.executeNextIteration()
        print(r.info)


if __name__ == "__main__":
    main()
