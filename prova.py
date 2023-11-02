from pek import IPEKPP, Dataset, metrics

"""
{'name': 'A1', 'n_clusters': 20}
{'name': 'A2', 'n_clusters': 35}
{'name': 'A3', 'n_clusters': 50}
...
{'name': 'Wine', 'n_clusters': 3}
"""
for d in Dataset.all():
    print(d)


dataset = Dataset("Wine")
km = IPEKPP(dataset.data, n_clusters=4, random_state=0)
while km.hasNextIteration():
    r = km.executeNextIteration()

    for metricName, metricFn in metrics.validation.all().items():
        r.metrics[metricName] = metricFn(dataset.data, r.labels)

    print(r.info, r.metrics)
