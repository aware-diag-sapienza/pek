import time

import numpy as np
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    InvalidParameterError,
    Real,
    StrOptions,
    validate_params,
)

from ..results.ensemble import (
    EnsemblePartialResult,
    EnsemblePartialResultInfo,
    EnsemblePartialResultMetrics,
)
from .interfaces import ProgressiveClusteringEnsemble
from .run import ProgressiveKMeansRun
from .termination import EarlyTerminationAction, EarlyTerminator
from .utils import adjustLabels, best_labels_dtype


def _checkEarlyTerminatorParamList(element):
    if element is None:
        return []
    elif isinstance(element, EarlyTerminator):
        return [element]
    elif isinstance(element, list):
        if all(isinstance(item, EarlyTerminator) for item in element):
            return element

    raise InvalidParameterError(f"The 'et' parameter must be instance of {EarlyTerminator.__class__} or a list of them")


class _InertiaBasedProgressiveEnsembleKMeans(ProgressiveClusteringEnsemble):
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "n_clusters": [Interval(Integral, 1, None, closed="left")],
            "n_runs": [Interval(Integral, 1, None, closed="left")],
            "init": [StrOptions({"k-means++", "random"})],
            "max_iter": [Interval(Integral, 1, None, closed="left")],
            "tol": [Interval(Real, 0, None, closed="left")],
            "random_state": ["random_state"],
            "freq": [None, Interval(Real, 0, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        X,
        n_clusters=2,
        n_runs=4,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=None,
        freq=None,
        et=None,
        minimizeLabelsChanging=True,
    ):
        self._X = X
        self._n_clusters = n_clusters
        self._n_runs = n_runs
        self._init = init
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._freq = freq
        self._earlyTerminatorList = _checkEarlyTerminatorParamList(et)
        self._minimizeLabelsChanging = minimizeLabelsChanging

        self._iteration = -1
        self._completed = False
        self._killed = False
        self._runs = []

        self._prevResult = None
        self._prevResultTimestamp = 0.0

        # create run objects
        for seed in np.random.default_rng(self._random_state).integers(0, np.iinfo(np.int32).max, size=self._n_runs):
            r = ProgressiveKMeansRun(
                self._X,
                n_clusters=self._n_clusters,
                max_iter=self._max_iter,
                tol=self._tol,
                random_state=seed,
                init=self._init,
            )
            self._runs.append(r)

        self._partitions = np.zeros((self._X.shape[0], self._n_runs), dtype=best_labels_dtype(self._n_clusters))
        self._centroids = np.zeros((self._n_clusters, self._X.shape[1], self._n_runs), dtype=float)
        self._runsLastPartialResultInfo = [None for _ in range(self._n_runs)]
        self._runsLastPartialResultMetrics = [None for _ in range(self._n_runs)]
        self._runsCompleted = [False for _ in range(self._n_runs)]
        self._runsKilled = [False for _ in range(self._n_runs)]
        self._runsInertia = [np.inf for _ in range(self._n_runs)]

    def hasNextIteration(self) -> bool:
        return not self._completed and not self._killed

    def executeNextIteration(self) -> EnsemblePartialResult:
        return self._executeNextIteration()

    def executeAllIterations(self) -> EnsemblePartialResult:
        r = None
        while self.hasNextIteration():
            r = self.executeNextIteration()
        return r

    def kill(self):
        self._killed = True

    def killRun(self, run):
        self._runsKilled[run] = True
        self._runs[run].kill()

    def _executeNextIteration(self) -> EnsemblePartialResult:
        if not self.hasNextIteration():
            raise RuntimeError("No next iteration to execute.")

        newComputation = [False for _ in range(self._n_runs)]
        iterationCost = 0
        for j in range(self._n_runs):
            if self._runs[j].hasNextIteration():
                iterationCost += 1
                newComputation[j] = True
                rp = self._runs[j].executeNextIteration()
                self._partitions[:, j] = rp.labels
                self._centroids[:, :, j] = rp.centroids
                self._runsLastPartialResultInfo[j] = rp.info
                self._runsLastPartialResultMetrics[j] = rp.metrics
                self._runsCompleted[j] = rp.info.isLast
                self._runsInertia[j] = rp.metrics.inertia

        self._iteration += 1
        self._completed = np.all([not self._runs[j].hasNextIteration() for j in range(self._n_runs)])

        bestRunIndex = int(np.argmin(self._runsInertia))
        worstRunIndex = int(np.argmax(self._runsInertia))
        bestCentroids = self._centroids[:, :, bestRunIndex]
        bestLabels = self._partitions[:, bestRunIndex]
        bestInertia = float(self._runsInertia[bestRunIndex])

        if newComputation[bestRunIndex] and self._minimizeLabelsChanging and self._prevResult is not None:
            self._partitions[:, bestRunIndex] = adjustLabels(bestLabels, bestCentroids, self._prevResult.centroids)
            bestLabels = self._partitions[:, bestRunIndex]

        runCompleted_str = "-".join(map(str, np.array(self._runsCompleted).astype(int)))
        runsKilled_str = "-".join(map(str, np.array(self._runsKilled).astype(int)))

        ensemblePartialResultInfo = EnsemblePartialResultInfo(
            self._iteration,
            self._completed,
            iterationCost,
            runCompleted_str,
            runsKilled_str,
            bestRun=bestRunIndex,
            worstRun=worstRunIndex,
        )

        ensemblePartialResultMetrics = EnsemblePartialResultMetrics(inertia=bestInertia)

        ensemblePartialResult = EnsemblePartialResult(
            info=ensemblePartialResultInfo,
            metrics=ensemblePartialResultMetrics,
            centroids=bestCentroids,
            labels=bestLabels,
            partitions=self._partitions,
            # self._runsLastPartialResultInfo,
            # self._runsLastPartialResultMetrics,
        )

        for et in self._earlyTerminatorList:
            action = et.checkEarlyTermination(ensemblePartialResult)
            if action == EarlyTerminationAction.NONE:
                continue
            elif action == EarlyTerminationAction.NOTIFY:
                ensemblePartialResult.setEarlyTermination(et.name, True)
            elif action == EarlyTerminationAction.KILL:
                ensemblePartialResult.setEarlyTermination(et.name, True)
                self.kill()

        # check results frequency
        currentTimestamp = time.time()
        elapsedFromPrevPartialResult = currentTimestamp - self._prevResultTimestamp
        if (self._freq is not None) and (elapsedFromPrevPartialResult < self._freq):
            time.sleep(self._freq - elapsedFromPrevPartialResult)

        self._prevResultTimestamp = time.time()
        self._prevResult = ensemblePartialResult

        return ensemblePartialResult


import inspect


class IPEK(_InertiaBasedProgressiveEnsembleKMeans):
    """Inertia-Based Progressive KMeans Clustering"""

    def __init__(
        self,
        X,
        n_clusters=2,
        n_runs=4,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        freq=None,
        et=None,
        minimizeLabelsChanging=True,
    ):
        kwargs = {k: v for k, v in filter(lambda d: d[0] not in ["self", "__class__"], locals().items())}

        del kwargs["X"]
        kwargs["init"] = "random"

        super().__init__(X, **kwargs)


class IPEKPP(_InertiaBasedProgressiveEnsembleKMeans):
    """Inertia-Based Progressive KMeans++ Clustering"""

    def __init__(
        self,
        X,
        n_clusters=2,
        n_runs=4,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        freq=None,
        et=None,
        minimizeLabelsChanging=True,
    ):
        kwargs = {k: v for k, v in filter(lambda d: d[0] not in ["self", "__class__"], locals().items())}

        del kwargs["X"]
        kwargs["init"] = "k-means++"

        super().__init__(X, **kwargs)
