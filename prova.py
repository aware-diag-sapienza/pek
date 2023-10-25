from pek import Dataset
from pek import IPEK, IPEKPP
import numpy as np

seed = 0
data = Dataset.load("Wine")


km = IPEKPP(data, n_clusters=4, random_state=seed)
while km.hasNextIteration():
    r = km.executeNextIteration()
    print(r.info, r.metrics)