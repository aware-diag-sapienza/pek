import time

import numpy as np
from sklearn.utils import Bunch
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    Real,
    StrOptions,
    validate_params,
)

from pek.termination.earlyTermination import EarlyTerminationAction

from ..utils.clustering import adjustLabels, best_labels_dtype
from ..utils.params import checkETS, checkValidationMetrics
from ..utils.random import get_random_state
from .run import ProgressiveKMeans


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
        )

    def setEarlyTermination(self, name, boolean):
        self.earlyTermination[name] = boolean


class EnsemblePartialResultInfo(Bunch):
    def __init__(self, iteration, seed, isLast, cost, runCompleted, runsKilled, bestRun):
        super().__init__(
            iteration=iteration,
            seed=seed,
            isLast=isLast,
            cost=cost,
            runCompleted=runCompleted,
            runsKilled=runsKilled,
            bestRun=bestRun,
        )


class EnsemblePartialResultMetrics(Bunch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class EnsemblePartialResultEarlyTermination(Bunch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ProgressiveEnsembleKMeans:
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
            "ets": [None, "array-like"],
            "validationMetrics": [None, "array-like"],
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
        freq=None,  # in frequency in seconds between results
        ets=None,  # early terminators (list)
        validationMetrics=None,
    ):
        self._X = X
        self._n_clusters = n_clusters
        self._n_runs = n_runs
        self._init = init
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = get_random_state(random_state)
        self._freq = freq
        self._ets = checkETS(ets)
        self._validationMetrics = checkValidationMetrics(validationMetrics)

        self._iteration = -1
        self._completed = False
        self._killed = False
        self._runs = []

        self._prevResult = None
        self._prevResultTimestamp = 0.0

        # create run objects
        for seed in np.random.default_rng(self._random_state).integers(0, np.iinfo(np.int32).max, size=self._n_runs):
            r = ProgressiveKMeans(
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

    def _executeNextIteration(self) -> EnsemblePartialResult:
        if not self.hasNextIteration():
            raise RuntimeError("No next iteration to execute.")

        # compute an iteration of each run
        iterationCost = 0
        for j in range(self._n_runs):
            if self._runs[j].hasNextIteration():
                iterationCost += 1
                rp = self._runs[j].executeNextIteration()
                self._partitions[:, j] = rp.labels
                self._centroids[:, :, j] = rp.centroids
                self._runsLastPartialResultInfo[j] = rp.info
                self._runsLastPartialResultMetrics[j] = rp.metrics
                self._runsCompleted[j] = rp.info.isLast
                self._runsInertia[j] = rp.metrics.inertia

        self._iteration += 1
        self._completed = np.all([not self._runs[j].hasNextIteration() for j in range(self._n_runs)])

        # choose the champion
        bestRunIndex = int(np.argmin(self._runsInertia))
        bestCentroids = self._centroids[:, :, bestRunIndex]
        bestLabels = self._partitions[:, bestRunIndex]
        bestInertia = float(self._runsInertia[bestRunIndex])

        # minimize label changing
        if self._prevResult is not None:
            self._partitions[:, bestRunIndex] = adjustLabels(bestLabels, bestCentroids, self._prevResult.centroids)
            bestLabels = self._partitions[:, bestRunIndex]

        # create the partial result (info)
        runCompleted_str = "-".join(map(str, np.array(self._runsCompleted).astype(int)))
        runsKilled_str = "-".join(map(str, np.array(self._runsKilled).astype(int)))
        ensemblePartialResultInfo = EnsemblePartialResultInfo(
            self._iteration,
            self._random_state,
            self._completed,
            iterationCost,
            runCompleted_str,
            runsKilled_str,
            bestRunIndex,
        )

        # create the partial result (metrics)
        metrics = {"inertia": bestInertia}
        for mName, mFunction in self._validationMetrics.items():
            if mName == "inertia":
                continue
            metrics[mName] = mFunction(self._X, bestLabels)

        ensemblePartialResultMetrics = EnsemblePartialResultMetrics(**metrics)

        # create the partial result
        ensemblePartialResult = EnsemblePartialResult(
            info=ensemblePartialResultInfo,
            metrics=ensemblePartialResultMetrics,
            centroids=bestCentroids,
            labels=bestLabels,
            partitions=self._partitions,
            # self._runsLastPartialResultInfo,
            # self._runsLastPartialResultMetrics,
        )

        # manage the early termination
        for et in self._ets:
            action = et.checkEarlyTermination(ensemblePartialResult)
            if action == EarlyTerminationAction.NONE:
                continue
            elif action == EarlyTerminationAction.NOTIFY:
                ensemblePartialResult.setEarlyTermination(et.name, True)
            elif action == EarlyTerminationAction.KILL:
                ensemblePartialResult.setEarlyTermination(et.name, True)
                self.kill()

        # manage results frequency
        currentTimestamp = time.time()
        elapsedFromPrevPartialResult = currentTimestamp - self._prevResultTimestamp
        if (self._freq is not None) and (elapsedFromPrevPartialResult < self._freq):
            time.sleep(self._freq - elapsedFromPrevPartialResult)

        # update previous result
        self._prevResultTimestamp = time.time()
        self._prevResult = ensemblePartialResult

        # return the current partial result
        return ensemblePartialResult

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
