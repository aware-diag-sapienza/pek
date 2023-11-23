import time

from pek import ProgressiveElbow, ProgressiveEnsembleKMeans
from pek.data import DatasetLoader


def main():
    dataset = DatasetLoader.load("LiverDisorder")

    km = ProgressiveEnsembleKMeans(
        dataset.data,
        n_clusters=6,
        n_runs=16,
        random_state=396350967,
        labelsValidationMetrics="ALL",
        labelsComparisonMetrics="ALL",
        labelsProgressionMetrics="ALL",
        partitionsValidationMetrics="ALL",
        partitionsComparisonMetrics="ALL",
        partitionsProgressionMetrics="ALL",
        ets=["slow-notify", "fast-notify"],
    )
    while km.hasNextIteration():
        r = km.executeNextIteration()
        print(r.info, r.earlyTermination)


if __name__ == "__main__":
    main()
