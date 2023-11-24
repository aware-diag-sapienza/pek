from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from umap import UMAP

d = {}

for file in sorted(Path("../csv").glob("*.csv"), key=lambda p: p.stem):
    name = file.stem.split("_")[0]
    n_clusters = int(file.stem.split("_")[1])
    d[name] = n_clusters

    df = pd.read_csv(file)

    # header
    header = np.asarray(list(df.columns), dtype=str, order="C")
    np.save(f"pek/data/_npy/{name}.header.npy", header)

    # npy
    arr = np.asarray(df.to_numpy(dtype=float), order="C")
    np.save(f"pek/data/_npy/{name}.npy", arr)

    # npyScaled
    arrScaled = np.asarray(StandardScaler().fit_transform(arr), order="C")
    np.save(f"pek/data/_npy/{name}.scaled.npy", arrScaled)

    ndim = arr.shape[1]

    # pca
    if ndim > 2:
        pca_proj = PCA(n_components=2, random_state=0).fit_transform(arrScaled)
        pca_proj = np.asarray(MinMaxScaler().fit_transform(pca_proj), order="C")
    else:
        pca_proj = arr
    np.save(f"pek/data/_npy/{name}.pca.npy", pca_proj)

    # tsne
    if ndim > 2:
        tsne_proj = TSNE(n_components=2, random_state=0).fit_transform(arrScaled)
        tsne_proj = np.asarray(MinMaxScaler().fit_transform(tsne_proj), order="C")
    else:
        tsne_proj = arr
    np.save(f"pek/data/_npy/{name}.tsne.npy", tsne_proj)

    # umap
    if ndim > 2:
        umap_proj = UMAP(random_state=0).fit_transform(arrScaled)
        umap_proj = np.asarray(MinMaxScaler().fit_transform(umap_proj), order="C")
    else:
        umap_proj = arr
    np.save(f"pek/data/_npy/{name}.umap.npy", umap_proj)

    print("done", name)


print(d)
