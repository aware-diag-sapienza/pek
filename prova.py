from pek import IPEKPP, Dataset, metrics

dataset = Dataset.get("Wine")
km = IPEKPP(dataset.data, n_clusters=4, random_state=0)
while km.hasNextIteration():
    r = km.executeNextIteration()

    for metricName, metricFn in metrics.validation.all().items():
        r.metrics[metricName] = metricFn(dataset.data, r.labels)

    print(r.info, r.metrics)
