from pek import metrics
from pek.clustering import (
    IPEK,
    IPEKPP,
    RatioInertiaEarlyTerminatorKiller,
    RatioInertiaEarlyTerminatorNotifier,
)
from pek.data import BuiltInDatasetLoader
from pek.process.ensemble import IPEK_Process
from pek.server import PEKServer

dataset = BuiltInDatasetLoader.load("Wine")
# print(dataset)

# server = PEKServer(1234)
# server.start()


IPEK(dataset.data, n_clusters=4, random_state=0, n_runs=5, et=RatioInertiaEarlyTerminatorKiller("slow", 10e-3))


"""km = IPEKPP(dataset.data, n_clusters=4, random_state=0, n_runs=5, et=RatioInertiaEarlyTerminatorKiller("slow", 10e-3))
while km.hasNextIteration():
    r = km.executeNextIteration()

    # if r.info.iteration == 4:
    #    km.killRun(3)

    # for metricName, metricFn in metrics.validation.all().items():
    #    r.metrics[metricName] = metricFn(dataset.data_scaled, r.labels)

    print(r.info, r.earlyTermination, r.metrics)"""

if __name__ == "__main__":
    # p = IPEK_Process(dataset.data, n_clusters=4, n_runs=16, minResultsFrequency=0)
    # p.start()
    # p.join()
    pass
