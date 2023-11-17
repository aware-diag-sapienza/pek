import numpy as np
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    InvalidParameterError,
    Real,
    StrOptions,
    validate_params,
)

from ..results import E
from .ensemble import _InertiaBasedProgressiveEnsembleKMeans
from .interfaces import ProgressiveClusteringEnsembleElbow
from .termination import EarlyTerminator


def _checkEarlyTerminatorParam(element):
    if element is None:
        return None
    elif isinstance(element, EarlyTerminator):
        return element

    raise InvalidParameterError(f"The 'et' parameter must be instance of {EarlyTerminator.__class__} or a list of them")


def _checkNClustersParam(arr):
    if isinstance(arr, list) or isinstance(arr, np.ndarray):
        try:
            int_arr = np.ndarray(arr, dtype=int)
            return int_arr
        except:
            pass
    raise InvalidParameterError(f"The 'et' parameter must be instance of list.")


class _InertiaBasedProgressiveEnsembleKMeansElbow(ProgressiveClusteringEnsembleElbow):
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            # "n_clusters": [Interval(Integral, 1, None, closed="left")],
            "n_runs": [Interval(Integral, 1, None, closed="left")],
            "init": [StrOptions({"k-means++", "random"})],
            "max_iter": [Interval(Integral, 1, None, closed="left")],
            "tol": [Interval(Real, 0, None, closed="left")],
            "random_state": ["random_state"],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        X,
        n_clusters,
        n_runs=4,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=None,
        et=None,
    ):
        self._X = X
        self._n_clusters = _checkNClustersParam(n_clusters)
        self._n_runs = n_runs
        self._init = init
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._et = _checkEarlyTerminatorParam(et)

        self._iteration = None
        self._completed = False
        self._killed = False
        self._ensembles = []

        for seed in np.random.default_rng(self._random_state).integers(
            0, np.iinfo(np.int32).max, size=len(self._n_clusters)
        ):
            e = _InertiaBasedProgressiveEnsembleKMeans(
                self._X,
                n_clusters=0,
                n_runs=self._n_runs,
                init=self._init,
                max_iter=self._max_iter,
                tol=self._tol,
                random_state=seed,
                minimizeLabelsChanging=False,
                et=self._et,
            )

            self._ensembles.append(e)

    def hasNextIteration(self) -> bool:
        return not self._completed and not self._killed

    def executeNextIteration(self) -> ElbowPartialResult:
        return self._executeNextIteration()

    def executeAllIterations(self) -> ElbowPartialResult:
        while self.hasNextIteration():
            r = self.executeNextIteration()
            if r.info.isLast:
                return r

    def kill(self):
        self._killed = True

    def _executeNextIteration(self) -> ElbowPartialResult:
        if not self.hasNextIteration():
            raise RuntimeError("No next iteration to execute.")

        e = self._ensembles.pop(0)
        ensemblePartialResult = e.executeAllIterations()

        self._completed = len(self._ensembles) == 0

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

        if self._iteration is None:
            self._iteration = 0
        else:
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

        self._prevResult = ensemblePartialResult
        return ensemblePartialResult
