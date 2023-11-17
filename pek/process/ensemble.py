import time
import uuid
from multiprocessing import Process, Queue

from ..clustering.ensemble import _InertiaBasedProgressiveEnsembleKMeans
from ..metrics import validation
from ..results.ensemble import EnsembleProcessPartialResult
from .utils import ProcessControlMessage, ProcessStatus


class EnsembleProcess(Process):
    def __init__(self, ensemble, minResultsFrequency=0, partialResultsQueue=None):
        super().__init__()
        self.id = str(uuid.uuid4())
        self._ensemble = ensemble
        self._status = ProcessStatus.PENDING
        self._partialResultsQueue = partialResultsQueue
        self._incomingControlMessagesQueue = Queue()
        self._lastPartialResultTimestamp = 0
        self._minResultsFrequency = minResultsFrequency

    def _readIncomingControlMessage(self, blocking=False):
        try:
            message = self._incomingControlMessagesQueue.get(block=blocking)
            print(message)

            if message == ProcessControlMessage.PAUSE:
                self._status = ProcessStatus.PAUSED
                self._readIncomingControlMessage(blocking=True)

            elif message == ProcessControlMessage.RESUME:
                self._status = ProcessStatus.RUNNING

            elif message == ProcessControlMessage.KILL:
                self._status = ProcessStatus.KILLED
                self._ensemble.kill()

        except:
            pass

    def _exec(self):
        while self._ensemble.hasNextIteration():
            self._readIncomingControlMessage(blocking=False)

            partialResult = self._ensemble.executeNextIteration()
            for metricName, metricFn in validation.all().items():
                partialResult.metrics[metricName] = metricFn(self._ensemble._X, partialResult.labels)

            currentTimestamp = time.time()
            elapsedFromLastPartialResult = currentTimestamp - self._lastPartialResultTimestamp
            if elapsedFromLastPartialResult < self._minResultsFrequency:
                delta = self._minResultsFrequency - elapsedFromLastPartialResult
                print("sleeping", delta)
                time.sleep(delta)
            self._lastPartialResultTimestamp = time.time()

            print(partialResult.info)

            if self._partialResultsQueue is not None:
                processPartialResult = EnsembleProcessPartialResult(self.id, partialResult)
                self._partialResultsQueue.put(processPartialResult)

        self._status = ProcessStatus.COMPLETED

    def start(self):
        self._exec()

    def pause(self):
        self._incomingControlMessagesQueue.put(ProcessControlMessage.PAUSE)

    def kill(self):
        self._incomingControlMessagesQueue.put(ProcessControlMessage.KILL)

    def resume(self):
        self._incomingControlMessagesQueue.put(ProcessControlMessage.RESUME)


class _InertiaBasedProgressiveEnsembleKMeansProcess(EnsembleProcess):
    def __init__(
        self,
        X,
        n_clusters=2,
        n_runs=4,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=None,
        minimizeLabelsChanging=True,
        et=None,
        minResultsFrequency=0,
        partialResultsQueue=None,
    ):
        super().__init__(
            _InertiaBasedProgressiveEnsembleKMeans(
                X,
                n_clusters=n_clusters,
                n_runs=n_runs,
                init=init,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state,
                minimizeLabelsChanging=minimizeLabelsChanging,
                et=et,
            ),
            minResultsFrequency=minResultsFrequency,
            partialResultsQueue=partialResultsQueue,
        )


class IPEK_Process(_InertiaBasedProgressiveEnsembleKMeansProcess):
    def __init__(
        self,
        X,
        n_clusters=2,
        n_runs=4,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        minimizeLabelsChanging=True,
        et=None,
        minResultsFrequency=0,
        partialResultsQueue=None,
    ):
        super().__init__(
            X,
            n_clusters=n_clusters,
            n_runs=n_runs,
            init="random",
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            minimizeLabelsChanging=minimizeLabelsChanging,
            et=et,
            minResultsFrequency=minResultsFrequency,
            partialResultsQueue=partialResultsQueue,
        )


class IPEKPP_Process(_InertiaBasedProgressiveEnsembleKMeansProcess):
    def __init__(
        self,
        X,
        n_clusters=2,
        n_runs=4,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        minimizeLabelsChanging=True,
        et=None,
        minResultsFrequency=0,
        partialResultsQueue=None,
    ):
        super().__init__(
            X,
            n_clusters=n_clusters,
            n_runs=n_runs,
            init="k-means++",
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            minimizeLabelsChanging=minimizeLabelsChanging,
            et=et,
            minResultsFrequency=minResultsFrequency,
            partialResultsQueue=partialResultsQueue,
        )
