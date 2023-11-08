from sklearn.utils import Bunch


class EnsemblePartialResult(Bunch):
    def __init__(self, info, metrics, centroids, labels, partitions, runsPartialResultInfo, runsPartialResultMetrics):
        if not isinstance(info, EnsemblePartialResultInfo):
            raise TypeError("info is not instance of EnsemblePartialResultInfo.")
        if not isinstance(metrics, EnsemblePartialResultMetrics):
            raise TypeError("metrics is not instance of EnsemblePartialResultMetrics.")
        super().__init__(
            info=info,
            metrics=metrics,
            centroids=centroids,
            labels=labels,
            partitions=partitions,
            runsPartialResultInfo=runsPartialResultInfo,
            runsPartialResultMetrics=runsPartialResultMetrics,
        )


class EnsemblePartialResultInfo(Bunch):
    def __init__(self, iteration, isLast, cost, runCompleted, **kwargs):
        super().__init__(iteration=iteration, isLast=isLast, cost=cost, runCompleted=runCompleted, **kwargs)


class EnsemblePartialResultMetrics(Bunch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


__all__ = [
    "EnsemblePartialResult",
    "EnsemblePartialResultInfo",
    "EnsemblePartialResultMetrics",
]
