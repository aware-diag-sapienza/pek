import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.cluster._k_means_lloyd import (
    lloyd_iter_chunked_dense,
    lloyd_iter_chunked_sparse,
)
from sklearn.cluster._kmeans import _BaseKMeans, _tolerance
from sklearn.utils import Bunch, check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    Real,
    StrOptions,
    validate_params,
)
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import threadpool_limits
from sklearn.utils.validation import _check_sample_weight

from ..metrics.validation import inertia as inertia_fn

# from ..utils.clustering import best_labels_dtype


class RunPartialResult(Bunch):
    def __init__(self, info, metrics, centroids, labels):
        if not isinstance(info, RunPartialResultInfo):
            raise TypeError("info is not instance of RunPartialResultInfo.")
        if not isinstance(metrics, RunPartialResultMetrics):
            raise TypeError("metrics is not instance of RunPartialResultMetrics.")
        super().__init__(info=info, metrics=metrics, centroids=centroids, labels=labels)


class RunPartialResultInfo(Bunch):
    def __init__(self, iteration, isLast):
        super().__init__(
            iteration=iteration,
            isLast=isLast,
        )


class RunPartialResultMetrics(Bunch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def _composePartialResult(iteration, completed, inertia, centroids, labels):
    info = RunPartialResultInfo(iteration, completed)
    metrics = RunPartialResultMetrics(inertia=inertia)
    return RunPartialResult(info, metrics, centroids, labels)


class ProgressiveKMeans(_BaseKMeans):
    """Progressive KMeans Algorithm. Using Lloyd Algorithm.\n
    Code edited from scikit-learn 1.3.0\n
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_kmeans.py

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    n_clusters : int, default=4
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    init : {'k-means++', 'random'}, Method for initialization, default=k-means++
    """

    @validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "n_clusters": [Interval(Integral, 1, None, closed="left")],
            "max_iter": [Interval(Integral, 1, None, closed="left")],
            "tol": [Interval(Real, 0, None, closed="left")],
            "random_state": ["random_state"],
            "init": [StrOptions({"k-means++", "random"})],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, X, n_clusters=4, max_iter=300, tol=1e-4, random_state=None, init="k-means++"):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init="warn",
            max_iter=max_iter,
            tol=tol,
            verbose=0,
            random_state=random_state,
        )

        self.X = self._validate_data(
            X, accept_sparse="csr", dtype=[np.float64, np.float32], order="C", copy=True, accept_large_sparse=False
        )

        random_state = check_random_state(self.random_state)
        x_squared_norms = row_norms(self.X, squared=True)
        centers_init = self._init_centroids(
            self.X,
            x_squared_norms=x_squared_norms,
            init=init,
            random_state=random_state,
            sample_weight=np.ones(self.X.shape[0], dtype=np.uint8),
        )

        if sp.issparse(X):
            self._iter_fn = lloyd_iter_chunked_sparse
            # self._inertia_fn = _inertia_sparse
        else:
            self._iter_fn = lloyd_iter_chunked_dense
            # self._inertia_fn = _inertia_dense

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = _tolerance(X, tol)
        self.random_state = random_state
        self.init = init

        self._killed = False
        self._completed = False
        self._converged = False
        self._convergedStrict = False

        self._iteration = None

        self._n_threads = _openmp_effective_n_threads()
        # Buffers to avoid new allocations at each iteration.
        self._centers = centers_init
        self._centers_new = np.zeros_like(self._centers)
        self._labels = np.full(self.X.shape[0], -1, dtype=np.int32)
        self._labels_old = self._labels.copy()
        self._weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
        self._center_shift = np.zeros(n_clusters, dtype=X.dtype)
        self._sample_weight = _check_sample_weight(None, X, dtype=X.dtype)

    def _warn_mkl_vcomp(self, n_active_threads):  # copied fron sklearn
        """Warn when vcomp and mkl are both present"""
        warnings.warn(
            "KMeans is known to have a memory leak on Windows "
            "with MKL, when there are less chunks than available "
            "threads. You can avoid it by setting the environment"
            f" variable OMP_NUM_THREADS={n_active_threads}."
        )

    def _executeNextIteration(self):
        if self._completed:
            raise RuntimeError("No next iteration to execute.")

        if self._iteration is None:
            with threadpool_limits(limits=1, user_api="blas"):
                self._iter_fn(
                    self.X,
                    self._sample_weight,
                    self._centers,
                    self._centers_new,
                    self._weight_in_clusters,
                    self._labels,
                    self._center_shift,
                    self._n_threads,
                    update_centers=False,
                )
                self._iteration = 0
                # inertia = self._inertia_fn(self.X, self._sample_weight, self._centers, self._labels, self._n_threads)
                inertia = inertia_fn(self.X, self._labels)
                self._completed = False
                return _composePartialResult(self._iteration, self._completed, inertia, self._centers, self._labels)

        if self._iteration == self.max_iter or self._converged:
            with threadpool_limits(limits=1, user_api="blas"):
                self._iter_fn(
                    self.X,
                    self._sample_weight,
                    self._centers,
                    self._centers_new,
                    self._weight_in_clusters,
                    self._labels,
                    self._center_shift,
                    self._n_threads,
                    update_centers=False,
                )
                self._iteration += 1
                # inertia = self._inertia_fn(self.X, self._sample_weight, self._centers, self._labels, self._n_threads)
                inertia = inertia_fn(self.X, self._labels)
                self._completed = True
                return _composePartialResult(self._iteration, self._completed, inertia, self._centers, self._labels)

        # Threadpoolctl context to limit the number of threads in second level of
        # nested parallelism (i.e. BLAS) to avoid oversubscription.
        with threadpool_limits(limits=1, user_api="blas"):
            self._iter_fn(
                self.X,
                self._sample_weight,
                self._centers,
                self._centers_new,
                self._weight_in_clusters,
                self._labels,
                self._center_shift,
                self._n_threads,
                update_centers=True,
            )
            self._iteration += 1
            # inertia = self._inertia_fn(self.X, self._sample_weight, self._centers, self._labels, self._n_threads)
            inertia = inertia_fn(self.X, self._labels)
            self._centers, self._centers_new = self._centers_new, self._centers

            if np.array_equal(self._labels, self._labels_old):
                # First check the labels for strict convergence.
                self._convergedStrict = True
                self._converged = True
                self._completed = True
            else:
                # No strict convergence, check for tol based convergence.
                center_shift_tot = (self._center_shift**2).sum()
                if center_shift_tot <= self.tol:
                    self._converged = True

            self._labels_old[:] = self._labels

            return _composePartialResult(self._iteration, self._completed, inertia, self._centers, self._labels)

    def hasNextIteration(self) -> bool:
        return not self._completed and not self._killed

    def executeNextIteration(self) -> RunPartialResult:
        return self._executeNextIteration()

    def executeAllIterations(self):
        r = None
        while self.hasNextIteration():
            r = self.executeNextIteration()
        return r

    def kill(self):
        self._killed = True
