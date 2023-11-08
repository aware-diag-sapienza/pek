from pek import metrics
from pek.clustering import IPEKPP
from pek.data import BuiltInDatasetLoader

dataset = BuiltInDatasetLoader.load("Wine")
print(dataset)


km = IPEKPP(dataset.data_scaled, n_clusters=4, random_state=0)
while km.hasNextIteration():
    r = km.executeNextIteration()

    for metricName, metricFn in metrics.validation.all().items():
        r.metrics[metricName] = metricFn(dataset.data_scaled, r.labels)

    print(r.info, r.metrics)
