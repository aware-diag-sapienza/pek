from sklearn.utils import Bunch


class RunPartialResult(Bunch):
    def __init__(self, info, metrics, labels):
        if not isinstance(info, RunPartialResultInfo):
            raise TypeError("info is not instance of RunPartialResultInfo.")
        if not isinstance(metrics, RunPartialResultMetrics):
            raise TypeError("metrics is not instance of RunPartialResultMetrics.")
        super().__init__(info=info, metrics=metrics, labels=labels)


class RunPartialResultInfo(Bunch):
    def __init__(self, iteration, isLast):
        super().__init__(
            iteration=iteration,
            isLast=isLast,
        )


class RunPartialResultMetrics(Bunch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class EnsemblePartialResult(Bunch):
    def __init__(self, info, metrics, labels, partitions, runsPartialResultInfo, runsPartialResultMetrics):
        if not isinstance(info, EnsemblePartialResultInfo):
            raise TypeError("info is not instance of EnsemblePartialResultInfo.")
        if not isinstance(metrics, EnsemblePartialResultMetrics):
            raise TypeError("metrics is not instance of EnsemblePartialResultMetrics.")
        super().__init__(
            info=info,
            metrics=metrics,
            labels=labels,
            partitions=partitions,
            runsPartialResultInfo=runsPartialResultInfo,
            runsPartialResultMetrics=runsPartialResultMetrics,
        )


class EnsemblePartialResultInfo(Bunch):
    def __init__(self, iteration, isLast, cost, runCompleted, **kwargs):
        super().__init__(iteration=iteration, isLast=isLast, cost=cost, runStatus=runCompleted, **kwargs)


class EnsemblePartialResultMetrics(Bunch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
