from pek import ProgressiveElbow
from pek.data import DatasetLoader
from pek.termination import EarlyTerminatorKiller

dataset = DatasetLoader.load("Wine")

eb = ProgressiveElbow(
    dataset.data, n_clusters_arr=list(range(2, 11)), n_runs=16, et=EarlyTerminatorKiller("fast", 1e-2)
)
while eb.hasNextIteration():
    r = eb.executeNextIteration()
    print(r.info, r.metrics)
