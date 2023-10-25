import numpy as np

from pek import IPEK, IPEKPP, Dataset, metrics

seed = 0
data = Dataset.load("Wine")


km = IPEKPP(data, n_clusters=4, random_state=seed)
while km.hasNextIteration():
    r = km.executeNextIteration()
    print(r.info, r.metrics)
