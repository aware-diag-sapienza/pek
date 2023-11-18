import warnings

import numpy as np
from kneed import KneeLocator
from sklearn.utils import Bunch
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    InvalidParameterError,
    Real,
    StrOptions,
    validate_params,
)

from ..utils.params import checkValidationMetrics
from ..utils.random import get_random_state
from .ensemble import ProgressiveEnsembleKMeans


class ElbowPartialResult(Bunch):
    def __init__(
        self, info=None, metrics=None, centroids=None, labels=None, partitions=None
    ):  # , runsPartialResultInfo, runsPartialResultMetrics
        if not isinstance(info, ElbowPartialResultInfo):
            raise TypeError("info is not instance of ElbowPartialResultInfo.")
        if not isinstance(metrics, ElbowPartialResultMetrics):
            raise TypeError("metrics is not instance of ElbowPartialResultMetrics.")
        super().__init__(
            info=info,
            metrics=metrics,
        )


class ElbowPartialResultInfo(Bunch):
    def __init__(self, seed, n_clusters, inertia, isLast, elbowPoint=None):
        super().__init__(seed=seed, n_clusters=n_clusters, inertia=inertia, isLast=isLast, elbowPoint=elbowPoint)


class ElbowPartialResultMetrics(Bunch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ProgressiveElbow:
    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "n_clusters_arr": [None, "array-like"],
            "n_runs": [Interval(Integral, 1, None, closed="left")],
            "init": [StrOptions({"k-means++", "random"})],
            "max_iter": [Interval(Integral, 1, None, closed="left")],
            "tol": [Interval(Real, 0, None, closed="left")],
            "random_state": ["random_state"],
            "validationMetrics": [None, "array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        X,
        n_clusters_arr=None,
        n_runs=4,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=None,
        et=None,  # early terminator (a single element or None),
        validationMetrics=None,
    ):
        self._X = X
        self._n_clusters_arr = sorted(n_clusters_arr)
        self._n_runs = n_runs
        self._init = init
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = get_random_state(random_state)
        self._ets = [] if et is None else [et]
        self._validationMetrics = checkValidationMetrics(validationMetrics)

        if n_clusters_arr is None:
            self._n_clusters_arr = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        if not all(isinstance(elem, int) for elem in n_clusters_arr):
            raise TypeError(f"The 'n_clusters_arr' parameter must contains only integers.")

        if len(self._n_clusters_arr) <= 2:
            raise InvalidParameterError(f"The 'n_clusters_arr' must have length >=2. Got {len(self._n_clusters_arr)}.")

        self._iteration = -1
        self._completed = False
        self._killed = False

        self._pending = [int(k) for k in self._n_clusters_arr]
        self._results = []

    def _executeNextIteration(self):
        if not self.hasNextIteration():
            raise RuntimeError("No next iteration to execute.")

        k = self._pending.pop(0)

        ensemble = ProgressiveEnsembleKMeans(
            self._X,
            n_clusters=k,
            n_runs=self._n_runs,
            init=self._init,
            max_iter=self._max_iter,
            tol=self._tol,
            random_state=self._random_state,
            ets=self._ets,
        )

        ensembleLastResult = ensemble.executeAllIterations()
        # add metrics
        for mName, mFunction in self._validationMetrics.items():
            if mName not in ensembleLastResult.metrics:
                ensembleLastResult.metrics[mName] = mFunction(self._X, ensembleLastResult.labels)

        self._iteration += 1
        self._completed = len(self._pending) == 0

        elbowResultInfo = ElbowPartialResultInfo(
            self._random_state, k, ensembleLastResult.metrics.inertia, self._completed
        )
        elbowResultMetrics = ElbowPartialResultMetrics(**ensembleLastResult.metrics)
        elbowResult = ElbowPartialResult(elbowResultInfo, elbowResultMetrics)

        self._results.append(elbowResult)

        # set the elbow value
        elbowResultInfo.elbowPoint = self._computeElbowPoint()

        return elbowResult

    def _computeElbowPoint(self):
        """Computes the elbow point using the inertia curve composed of all the past partial results.
        Returns the n_cluster value of the elbow, if exists. Otherwise, returns None."""
        inertiaCurve = np.array([[r.info.n_clusters, r.info.inertia] for r in self._results])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                kneedle = KneeLocator(
                    np.array(inertiaCurve)[:, 0],
                    np.array(inertiaCurve)[:, 1],
                    S=1.0,
                    curve="convex",
                    direction="decreasing",
                )
                return int(kneedle.elbow)
        except:
            return None

    def hasNextIteration(self) -> bool:
        return not self._completed and not self._killed

    def executeNextIteration(self) -> ElbowPartialResult:
        return self._executeNextIteration()

    def executeAllIterations(self) -> ElbowPartialResult:
        r = None
        while self.hasNextIteration():
            r = self.executeNextIteration()
        return r

    def kill(self):
        self._killed = True
