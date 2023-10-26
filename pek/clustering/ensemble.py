import numpy as np
from sklearn.utils._param_validation import (
    Integral,
    Interval,
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
from .utils import best_labels_dtype


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
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, X, n_clusters=2, n_runs=4, init="k-means++", max_iter=300, tol=1e-4, random_state=None):
        self._X = X
        self._n_clusters = n_clusters
        self._n_runs = n_runs
        self._init = init
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state

        self._iteration = None
        self._completed = False
        self._runs = []

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
        self._runsLastPartialResultInfo = [None for _ in range(self._n_runs)]
        self._runsLastPartialResultMetrics = [None for _ in range(self._n_runs)]
        self._runsCompleted = [False for _ in range(self._n_runs)]
        self._runsInertia = [np.inf for _ in range(self._n_runs)]

    def hasNextIteration(self) -> bool:
        return not self._completed

    def executeNextIteration(self) -> EnsemblePartialResult:
        return self._executeNextIteration()

    def _executeNextIteration(self) -> EnsemblePartialResult:
        if not self.hasNextIteration():
            raise RuntimeError("No next iteration to execute.")

        iterationCost = 0
        for j in range(self._n_runs):
            if self._runs[j].hasNextIteration():
                iterationCost += 1
                rp = self._runs[j].executeNextIteration()
                self._partitions[:, j] = rp.labels
                self._runsLastPartialResultInfo[j] = rp.info
                self._runsLastPartialResultMetrics[j] = rp.metrics
                self._runsCompleted[j] = rp.info.isLast
                self._runsInertia[j] = rp.metrics.inertia

        if self._iteration is None:
            self._iteration = 0
        else:
            self._iteration += 1

        self._completed = np.all(self._runsCompleted)
        bestRunIndex = int(np.argmin(self._runsInertia))
        worstRunIndex = int(np.argmax(self._runsInertia))
        bestLabels = self._partitions[:, bestRunIndex]
        bestInertia = float(self._runsInertia[bestRunIndex])

        runCompleted_str = "-".join(map(str, np.array(self._runsCompleted).astype(int)))
        ensemblePartialResultInfo = EnsemblePartialResultInfo(
            self._iteration,
            self._completed,
            iterationCost,
            runCompleted_str,
            bestRun=bestRunIndex,
            worstRun=worstRunIndex,
        )
        ensemblePartialResultMetrics = EnsemblePartialResultMetrics(inertia=bestInertia)
        ensemblePartialResult = EnsemblePartialResult(
            ensemblePartialResultInfo,
            ensemblePartialResultMetrics,
            bestLabels,
            self._partitions,
            self._runsLastPartialResultInfo,
            self._runsLastPartialResultMetrics,
        )
        return ensemblePartialResult


class IPEK(_InertiaBasedProgressiveEnsembleKMeans):
    """Inertia-Based Progressive KMeans Clustering"""

    def __init__(self, X, n_clusters=2, n_runs=4, max_iter=300, tol=1e-4, random_state=None):
        super().__init__(
            X,
            n_clusters=n_clusters,
            n_runs=n_runs,
            init="random",
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )


class IPEKPP(_InertiaBasedProgressiveEnsembleKMeans):
    """Inertia-Based Progressive KMeans++ Clustering"""

    def __init__(self, X, n_clusters=2, n_runs=4, max_iter=300, tol=1e-4, random_state=None):
        super().__init__(
            X,
            n_clusters=n_clusters,
            n_runs=n_runs,
            init="k-means++",
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
