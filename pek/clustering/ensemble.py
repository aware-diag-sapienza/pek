import time
from abc import ABC
from multiprocessing import Process, Queue

import numpy as np
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    Real,
    StrOptions,
    validate_params,
)

from pek.termination.earlyTermination import EarlyTerminationAction

from ..metrics.comparison import _toComparisonMetricDict
from ..metrics.progression import _toProgressionMetricDict
from ..metrics.validation import _toValidationMetricDict
from ..termination.earlyTermination import _check_et_list
from ..utils.clustering import adjustLabels  # , best_labels_dtype
from ..utils.params import checkInstance
from ..utils.process import (
    ProcessControlMessage,
    ProcessControlMessageType,
    ProcessStatus,
)
from ..utils.random import get_random_state
from .results import (
    EnsemblePartialResult,
    EnsemblePartialResultEarlyTermination,
    EnsemblePartialResultInfo,
    EnsemblePartialResultMetrics,
    EnsemblePartialResultRunsStatus,
    MetricGroup,
)
from .run import ProgressiveKMeans


class _AbstractProgressiveEnsembleKMeans(ABC):
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
            # "labelsValidationMetrics": [None, str, "array-like"],
            # "labelsComparisonMetrics": [None, str, "array-like"],
            # "labelsProgressionMetrics": [None, str, "array-like"],
            # "partitionsValidationMetrics": [None, str, "array-like"],
            # "partitionsComparisonMetrics": [None, str, "array-like"],
            # "partitionsProgressionMetrics": [None, str, "array-like"],
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
        labelsValidationMetrics=None,
        labelsComparisonMetrics=None,
        labelsProgressionMetrics=None,
        partitionsValidationMetrics=None,
        partitionsComparisonMetrics=None,
        partitionsProgressionMetrics=None,
        taskId=None,
    ):
        self._X = X
        self._n_clusters = n_clusters
        self._n_runs = n_runs
        self._init = init
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = get_random_state(random_state)
        self._freq = freq
        self._ets = _check_et_list(ets)
        self._metricsCalculator = _EnsembleMetricsCalculator(
            X,
            labelsValidationMetrics=labelsValidationMetrics,
            labelsComparisonMetrics=labelsComparisonMetrics,
            labelsProgressionMetrics=labelsProgressionMetrics,
            partitionsValidationMetrics=partitionsValidationMetrics,
            partitionsComparisonMetrics=partitionsComparisonMetrics,
            partitionsProgressionMetrics=partitionsProgressionMetrics,
        )
        self._taskId = taskId


class ProgressiveEnsembleKMeans(_AbstractProgressiveEnsembleKMeans):
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
        ets=None,
        labelsValidationMetrics=None,
        labelsComparisonMetrics=None,
        labelsProgressionMetrics=None,
        partitionsValidationMetrics=None,
        partitionsComparisonMetrics=None,
        partitionsProgressionMetrics=None,
        taskId=None,
    ):
        super().__init__(
            X,
            n_clusters=n_clusters,
            n_runs=n_runs,
            init=init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            freq=freq,
            ets=ets,
            labelsValidationMetrics=labelsValidationMetrics,
            labelsComparisonMetrics=labelsComparisonMetrics,
            labelsProgressionMetrics=labelsProgressionMetrics,
            partitionsValidationMetrics=partitionsValidationMetrics,
            partitionsComparisonMetrics=partitionsComparisonMetrics,
            partitionsProgressionMetrics=partitionsProgressionMetrics,
            taskId=taskId,
        )

        self._iteration = -1
        self._completed = False
        self._killed = False
        self._runs = []

        self._prevResultCentroids = None
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

        self._partitions = np.zeros((self._n_runs, self._X.shape[0]), dtype=int)
        self._centroids = np.zeros((self._n_clusters, self._X.shape[1], self._n_runs), dtype=float)
        self._runsLastPartialResultInfo = [None for _ in range(self._n_runs)]
        self._runsLastPartialResultMetrics = [None for _ in range(self._n_runs)]
        self._runsIteration = [None for _ in range(self._n_runs)]
        self._runsCompleted = [False for _ in range(self._n_runs)]
        self._runsKilled = [False for _ in range(self._n_runs)]
        self._runsInertia = [np.inf for _ in range(self._n_runs)]

        self._disabledEts = [False for _ in self._ets]

    def _executeNextIteration(self) -> EnsemblePartialResult:
        if not self.hasNextIteration():
            raise RuntimeError("No next iteration to execute.")

        # compute an iteration of each run
        iterationCost = 0
        for i in range(self._n_runs):
            if self._runs[i].hasNextIteration():
                iterationCost += 1
                rp = self._runs[i].executeNextIteration()
                self._partitions[i, :] = rp.labels
                self._centroids[:, :, i] = rp.centroids
                self._runsLastPartialResultInfo[i] = rp.info
                self._runsLastPartialResultMetrics[i] = rp.metrics
                self._runsCompleted[i] = rp.info.isLast
                self._runsInertia[i] = rp.metrics.inertia
                self._runsIteration[i] = rp.info.iteration

        self._iteration += 1
        self._completed = np.all([not self._runs[j].hasNextIteration() for j in range(self._n_runs)])

        # choose the champion
        bestRunIndex = int(np.argmin(self._runsInertia))
        bestCentroids = self._centroids[:, :, bestRunIndex]
        bestLabels = self._partitions[bestRunIndex, :]
        bestInertia = float(self._runsInertia[bestRunIndex])

        # minimize label changing
        if self._prevResultCentroids is not None:
            self._partitions[bestRunIndex, :] = adjustLabels(bestLabels, bestCentroids, self._prevResultCentroids)
            bestLabels = self._partitions[bestRunIndex, :]

        # create the partial result (info)
        last = not self.hasNextIteration()

        ensemblePartialResultInfo = EnsemblePartialResultInfo(
            self._iteration,
            self._random_state,
            last,
            self._completed,
            iterationCost,
            bestRunIndex,
            bestInertia,
        )

        # runsStatus
        runIteration_str = "-".join(map(str, np.array(self._runsIteration).astype(int)))
        runCompleted_str = "-".join(map(str, np.array(self._runsCompleted).astype(int)))
        runsKilled_str = "-".join(map(str, np.array(self._runsKilled).astype(int)))
        runsStatus = EnsemblePartialResultRunsStatus(
            runIteration=runIteration_str, runCompleted=runCompleted_str, runsKilled=runsKilled_str
        )

        # create the partial result (metrics)
        metrics = self._metricsCalculator.getMetrics(bestRunIndex, self._runsInertia, self._centroids, self._partitions)

        # create the partial result
        ensemblePartialResult = EnsemblePartialResult(
            info=ensemblePartialResultInfo,
            metrics=metrics,
            centroids=bestCentroids,
            labels=bestLabels,
            partitions=self._partitions,
            runsStatus=runsStatus,
            taskId=self._taskId,
        )

        # manage the early termination
        for i, et in enumerate(self._ets):
            if self._disabledEts[i]:
                continue
            action = et.checkEarlyTermination(ensemblePartialResult)
            if action == EarlyTerminationAction.NONE:
                continue
            elif action == EarlyTerminationAction.NOTIFY:
                self._disabledEts[i] = True
                ensemblePartialResult._setEarlyTermination(et.name, True)
            elif action == EarlyTerminationAction.KILL:
                self._disabledEts[i] = True
                ensemblePartialResult._setEarlyTermination(et.name, True)
                ensemblePartialResult.info.last = True
                self.kill()

        # manage results frequency
        currentTimestamp = time.time()
        elapsedFromPrevPartialResult = currentTimestamp - self._prevResultTimestamp
        if (self._freq is not None) and (elapsedFromPrevPartialResult < self._freq):
            time.sleep(self._freq - elapsedFromPrevPartialResult)

        # update previous result
        self._prevResultCentroids = bestCentroids
        self._prevResultTimestamp = time.time()

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


class _EnsembleMetricsCalculator:
    def __init__(
        self,
        X,
        labelsValidationMetrics=None,
        labelsComparisonMetrics=None,
        labelsProgressionMetrics=None,
        partitionsValidationMetrics=None,
        partitionsComparisonMetrics=None,
        partitionsProgressionMetrics=None,
    ):
        self._X = X
        self._labelsValidationMetrics = _toValidationMetricDict(labelsValidationMetrics)
        self._labelsComparisonMetrics = _toComparisonMetricDict(labelsComparisonMetrics)
        self._labelsProgressionMetrics = _toProgressionMetricDict(labelsProgressionMetrics)

        self._partitionsValidationMetrics = _toValidationMetricDict(partitionsValidationMetrics)
        self._partitionsComparisonMetrics = _toComparisonMetricDict(partitionsComparisonMetrics)
        self._partitionsProgressionMetrics = _toProgressionMetricDict(partitionsProgressionMetrics)

        self._bestLabelsPrev = None
        self._labelsHistory = []
        self._partitionsHistory = []

    def getMetrics(self, bestRunIndex, runsInertia, centroids, partitions) -> EnsemblePartialResultMetrics:
        return EnsemblePartialResultMetrics(
            labelsValidationMetrics=self._compute_labelsValidationMetrics(
                bestRunIndex, runsInertia, centroids, partitions
            ),
            labelsComparisonMetrics=self._compute_labelsComparisonMetrics(
                bestRunIndex, runsInertia, centroids, partitions
            ),
            labelsProgressionMetrics=self._compute_labelsProgressionMetrics(
                bestRunIndex, runsInertia, centroids, partitions
            ),
            partitionsValidationMetrics=self._compute_partitionsValidationMetrics(
                bestRunIndex, runsInertia, centroids, partitions
            ),
            partitionsComparisonMetrics=self._compute_partitionsComparisonMetrics(
                bestRunIndex, runsInertia, centroids, partitions
            ),
            partitionsProgressionMetrics=self._compute_partitionsProgressionMetrics(
                bestRunIndex, runsInertia, centroids, partitions
            ),
        )

    def _compute_labelsValidationMetrics(self, bestRunIndex, runsInertia, centroids, partitions):
        """Labels validation metrics are computed only on the current best labels."""
        bestInertia = float(runsInertia[bestRunIndex])
        bestLabels = partitions[bestRunIndex, :]

        res = {"inertia": bestInertia}
        for metricName, metricFunction in self._labelsValidationMetrics.items():
            if metricName not in res:
                res[metricName] = metricFunction(self._X, bestLabels)

        return MetricGroup(**res)

    def _compute_labelsComparisonMetrics(self, bestRunIndex, runsInertia, centroids, partitions):
        """Labels comparison metrics are computed comparing the current best labels with the previous best labels.
        The initial iteration has np.nan"""
        bestLabels = partitions[bestRunIndex, :]

        res = {}
        for metricName, metricFunction in self._labelsComparisonMetrics.items():
            if metricName not in res:
                if self._bestLabelsPrev is None:
                    res[metricName] = None
                else:
                    res[metricName] = metricFunction(bestLabels, self._bestLabelsPrev)

        if len(self._labelsComparisonMetrics) > 0:
            self._bestLabelsPrev = bestLabels

        return MetricGroup(**res)

    def _compute_labelsProgressionMetrics(self, bestRunIndex, runsInertia, centroids, partitions):
        if len(self._labelsProgressionMetrics) > 0:
            self._labelsHistory.append(partitions[bestRunIndex, :])

        res = {}
        for metricName, metricFunction in self._labelsProgressionMetrics.items():
            if metricName not in res:
                if len(self._labelsHistory) == 1:
                    res[metricName] = None
                else:
                    res[metricName] = metricFunction(self._labelsHistory)

        return MetricGroup(**res)

    def _compute_partitionsValidationMetrics(self, bestRunIndex, runsInertia, centroids, partitions):
        """Partitions validation metrics are computed on each partition.
        The result is a dictionary where each metric has an array of values, one for each partition.
        """
        res = {"inertia": runsInertia}
        for metricName, metricFunction in self._partitionsValidationMetrics.items():
            if metricName not in res:
                res[metricName] = np.empty(partitions.shape[0], dtype=float)
                for i in range(partitions.shape[0]):
                    res[metricName][i] = metricFunction(self._X, partitions[i, :])

        return MetricGroup(**res)

    def _compute_partitionsComparisonMetrics(self, bestRunIndex, runsInertia, centroids, partitions):
        """
        Partitions comparison metrics are computed on each pair of partition.
        The result is a dictionary where each metric has a symmetric matrix RxR.
        """
        n_runs = partitions.shape[0]
        res = {}
        for metricName, metricFunction in self._partitionsComparisonMetrics.items():
            if metricName not in res:
                res[metricName] = np.empty((n_runs, n_runs), dtype=float)

                for i in range(n_runs):
                    for j in range(n_runs):
                        if j >= i:
                            continue
                        val = metricFunction(partitions[i, :], partitions[j, :])
                        res[metricName][i, j] = val
                        res[metricName][j, i] = val

        return MetricGroup(**res)

    def _compute_partitionsProgressionMetrics(self, bestRunIndex, runsInertia, centroids, partitions):
        if len(self._partitionsProgressionMetrics) > 0:
            self._partitionsHistory.append(partitions)

        n_runs = partitions.shape[0]
        res = {}
        for metricName, metricFunction in self._partitionsProgressionMetrics.items():
            if metricName not in res:
                res[metricName] = [None for j in range(n_runs)]
                for i in range(n_runs):
                    if len(self._partitionsHistory) > 1:
                        hist = [p[i, :] for p in self._partitionsHistory]
                        res[metricName][i] = metricFunction(hist)

        return MetricGroup(**res)


class ProgressiveEnsembleKMeansProcess(Process):
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
        ets=None,
        labelsValidationMetrics=None,
        labelsComparisonMetrics=None,
        labelsProgressionMetrics=None,
        partitionsValidationMetrics=None,
        partitionsComparisonMetrics=None,
        partitionsProgressionMetrics=None,
        taskId=None,
        verbose=False,
        resultsQueue=None,
        **wkargs,
    ):
        super().__init__()

        self._ensemble = ProgressiveEnsembleKMeans(
            X,
            n_clusters=n_clusters,
            n_runs=n_runs,
            init=init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            freq=freq,
            ets=ets,
            labelsValidationMetrics=labelsValidationMetrics,
            labelsComparisonMetrics=labelsComparisonMetrics,
            labelsProgressionMetrics=labelsProgressionMetrics,
            partitionsValidationMetrics=partitionsValidationMetrics,
            partitionsComparisonMetrics=partitionsComparisonMetrics,
            partitionsProgressionMetrics=partitionsProgressionMetrics,
            taskId=taskId,
        )

        self._verbose = verbose
        self._status = ProcessStatus.PENDING
        self._resultsQueue = resultsQueue
        self._controlsQueue = Queue()

    def _waitForResume(self):
        while True:
            msg = self._controlsQueue.get(block=True)
            if (
                msg.messageType == ProcessControlMessageType.RESUME
                or msg.messageType == ProcessControlMessageType.START
            ):
                self._status = ProcessStatus.RUNNING
                return

    def _readControlMessage(self):
        try:
            msg = self._controlsQueue.get(block=False)

            if msg.messageType == ProcessControlMessageType.PAUSE:
                self._status = ProcessStatus.PAUSED
                self._waitForResume()

            elif msg.messageType == ProcessControlMessageType.KILL_RUN:
                runId = msg.messageData.runId
                self._ensemble.killRun(runId)

            elif msg.messageType == ProcessControlMessageType.KILL:
                self._status = ProcessStatus.KILLED
                self._ensemble.kill()
        except:
            pass

    def run(self):
        self._status = ProcessStatus.RUNNING

        while self._ensemble.hasNextIteration():
            r = self._ensemble.executeNextIteration()
            if self.resultQueue is not None:
                self._resultsQueue.put(r)
            if self._verbose:
                print(r.info)

            if self._ensemble.hasNextIteration():
                self._readControlMessage()

        self._status = ProcessStatus.COMPLETED
        exit()

    @property
    def controlsQueue(self):
        return self._controlsQueue

    @property
    def resultQueue(self):
        return self._resultsQueue

    def pause(self):
        msg = ProcessControlMessage.PAUSE()
        self._controlsQueue.put(msg)

    def resume(self):
        msg = ProcessControlMessage.RESUME()
        self._controlsQueue.put(msg)

    def kill(self):
        msg = ProcessControlMessage.KILL()
        self._controlsQueue.put(msg)

    def killRun(self, runId):
        msg = ProcessControlMessage.KILL_RUN(runId)
        self._controlsQueue.put(msg)
