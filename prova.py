import numpy as np

from pek import IPEK, IPEKPP, Dataset, metrics

seed = 0
data = Dataset.load("Wine")


km = IPEKPP(data, n_clusters=4, random_state=seed)
while km.hasNextIteration():
    r = km.executeNextIteration()

    for metricName, metricFn in metrics.validation.all().items():
        r.metrics[metricName] = metricFn(data, r.labels)

    print(r.info, r.metrics)
