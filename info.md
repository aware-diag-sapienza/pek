# pek
 Progressive ensemble k-means clustering

## Installation
```bash
pip3 install pek
```

## Server
To run pek the server:
```bash
python3 -m pek.server [-p <port>]
```

## Client JavaScript Library
### Ensemble Task
- `dataset`: Name of the dataset. Error if not passed.
- `n_clusters`: Integer. Default 2.
- `n_runs`: Number of runs. Default 4.
- `init`: Initialization algorithm in {'k-means++', 'random'}. Default 'k-means++'.
- `max_iter`: Maximum number of iterations. Default 300.
- `tol`: Tolerance for centroids convergence. Default 1e-4.
- `random_state`: Integer for seeding. Default null.
- `freq`: Min number of seconds (float) before producing new partial result. Default null.
- `ets`: List of early terminators. Select objects/strings from the available choices.
- `labelsValidationMetrics` Array of validation metrics for the labels to compute. Pass the string "ALL" instead of the array to have all metrics. Default null, or empty array.,
- `labelsComparisonMetrics` Array of comparison metrics for the labels to compute. Pass the string "ALL" instead of the array to have all metrics. Default null, or empty array.
- `labelsProgressionMetrics` Array of progression metrics for the labels to compute. Pass the string "ALL" instead of the array to have all metrics. Default null, or empty array.
- `partitionsValidationMetrics` Array of validation metrics for partitions to compute. Pass the string "ALL" instead of the array to have all metrics. Default null, or empty array.
- `partitionsComparisonMetrics` Array of comparison metrics for partitions to compute. Pass the string "ALL" instead of the array to have all metrics. Default null, or empty array.
- `partitionsProgressionMetrics`Array of progression metrics for partitions to compute. Pass the string "ALL" instead of the array to have all metrics. Default null, or empty array.



### Elbow Task
- `dataset`: Name of the dataset. Error if not passed.
- `n_clusters_arr`: Array of integers of k to compute. Default [2, 3, ..., 10].
- `n_runs`: Number of runs. Default 4.
- `init`: Initialization algorithm in {'k-means++', 'random'}. Default 'k-means++'.
- `max_iter`: Maximum number of iterations. Default 300.
- `tol`: Tolerance for centroids convergence. Default 1e-4.
- `random_state`: Integer for seeding. Default null.
- `freq`: Min number of seconds (float) before producing new partial result. Default null.
- `validationMetrics`: Array of validation metrics to compute. Pass the string "ALL" instead of the array to have all metrics. Default null, or empty array.
- `et`: Early termination. A single object/string from the available choices.


### Default Parameters

#### Early Terminators
An `et` can be represented as:
- a standard ET encoded as string from {fast-notify, fast-kill, slow-notify, slow-kill}
- a custom ET as dictionary with `{name: '...', threshold: x, action: '...' }`. The action can be either `notify` ot `kill`.

#### Validation Metrics
- calinski_harabasz
- davies_bouldin
- dunn_index
- inertia
- silhouette
- simplified_silhouette

#### Comparison Metrics
- ari
- ami

### Progression Metrics
- entries_stability_2
- entries_stability_3
- entries_stability_5
- entries_stability_10
- entries_stability_all


- global_stability_2
- global_stability_3
- global_stability_5
- global_stability_10
- global_stability_all
