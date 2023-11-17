from sklearn.utils import Bunch


class EnsemblePartialResult(Bunch):
    def __init__(
        self, info=None, metrics=None, centroids=None, labels=None, partitions=None
    ):  # , runsPartialResultInfo, runsPartialResultMetrics
        if not isinstance(info, EnsemblePartialResultInfo):
            raise TypeError("info is not instance of EnsemblePartialResultInfo.")
        if not isinstance(metrics, EnsemblePartialResultMetrics):
            raise TypeError("metrics is not instance of EnsemblePartialResultMetrics.")
        super().__init__(
            info=info,
            earlyTermination=EnsemblePartialResultEarlyTermination(),
            metrics=metrics,
            centroids=centroids,
            labels=labels,
            partitions=partitions,
            # runsPartialResultInfo=runsPartialResultInfo,
            # runsPartialResultMetrics=runsPartialResultMetrics,
        )

    def setEarlyTermination(self, name, boolean):
        self.earlyTermination[name] = boolean


class EnsemblePartialResultInfo(Bunch):
    def __init__(self, iteration, isLast, cost, runCompleted, runsKilled, **kwargs):
        super().__init__(
            iteration=iteration, isLast=isLast, cost=cost, runCompleted=runCompleted, runsKilled=runsKilled, **kwargs
        )


class EnsemblePartialResultMetrics(Bunch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class EnsemblePartialResultEarlyTermination(Bunch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class EnsembleProcessPartialResult(Bunch):
    def __init__(self, id, partialResult):
        super().__init__(id=id, partialResult=partialResult)


__all__ = [
    "EnsemblePartialResult",
    "EnsemblePartialResultInfo",
    "EnsemblePartialResultMetrics",
    "EnsemblePartialResultEarlyTermination",
    "EnsembleProcessPartialResult",
]
