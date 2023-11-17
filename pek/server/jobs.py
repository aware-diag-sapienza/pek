import math
import time
import uuid
from abc import ABC
from multiprocessing import Process

from sklearn.utils import Bunch
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    InvalidParameterError,
    Real,
    StrOptions,
    validate_params,
)

from ..clustering import IPEK, IPEKPP
from ..data import BuiltInDatasetLoader
from ..metrics import validation


class _Job(ABC):
    def start(self):
        pass

    def stop(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass


class EnsembleJob(Process):
    @validate_params(
        {
            "type": [StrOptions({"IPEK", "IPEKPP"})],
            "n_clusters": [Interval(Integral, 2, None, closed="left")],
            "n_runs": [Interval(Integral, 2, None, closed="left")],
            "random_state": ["random_state"],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        type,
        dataset,
        n_clusters,
        n_runs,
        random_state,
        partialResultsQueue,
        client=None,
        resultsMinFreq=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id = str(uuid.uuid4())

        self.type = type
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.n_runs = n_runs
        self.random_state = random_state
        self.partialResultsQueue = partialResultsQueue
        self.client = client
        self.resultsMinFreq = resultsMinFreq

        self.lastPartialResultTimestamp = 0

    def start(self):
        dataset = BuiltInDatasetLoader.load(self.dataset)
        ensemble = None
        if self.type == "IPEK":
            ensemble = IPEK(
                dataset.data, n_clusters=self.n_clusters, n_runs=self.n_runs, random_state=self.random_state
            )
        elif self.type == "IPEKPP":
            ensemble = IPEKPP(
                dataset.data, n_clusters=self.n_clusters, n_runs=self.n_runs, random_state=self.random_state
            )
        else:
            raise TypeError(f"The type '{self.type}' is not valid.")

        while ensemble.hasNextIteration():
            r = ensemble.executeNextIteration()
            for metricName, metricFn in validation.all().items():
                r.metrics[metricName] = metricFn(dataset.data_scaled, r.labels)
            self.partialResultsQueue.put(
                Bunch(jobId=self.id, jobClass="EnsembleJob", client=self.client, partialResult=r)
            )

            currentTimestamp = time.time()
            if self.resultsMinFreq is not None:
                delta = currentTimestamp - self.lastPartialResultTimestamp - self.resultsMinFreq
                delta = max(0, delta)
                time.sleep(delta)

            self.lastPartialResultTimestamp = currentTimestamp


class ElbowJob(Process):
    def __init__(
        self,
        type,
        dataset,
        min_n_clusters,
        max_n_clusters,
        n_runs,
        random_state,
        partialResultsQueue,
        client=None,
        earlyTermination=None,
        resultsMinFreq=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # self.id = f"ElbowJob:{dataset}-k[{min_n_clusters}:{max_n_clusters}]-r{n_runs}-s{random_state}-t{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-c{client}--et[{earlyTermination}]"
        self.id = str(uuid.uuid4())

        self.type = type
        self.dataset = dataset
        self.n_runs = n_runs
        self.min_n_clusters = min_n_clusters
        self.max_n_clusters = max_n_clusters
        self.k_values = list(range(self.min_n_clusters, self.max_n_clusters + 1))
        self.random_state = random_state  # elbow random state
        self.client = client
        self.earlyTermination = earlyTermination
        self.partialResultsQueue = partialResultsQueue
        self.resultsMinFreq = resultsMinFreq

        """self.data = Dataset(self.dataset).data()
        self.timeManager = None
        self.currentPec = None
        self.isLastK = False
        self.inertiaCurve = []
        self.cacheManager = CacheManager(subfolder='elbow')
        """
