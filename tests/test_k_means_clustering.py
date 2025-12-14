import sys
from pathlib import Path

import numpy as np

root = Path().resolve()
while not (root / "src" / "rice_ml").exists() and root != root.parent:
    root = root.parent
sys.path.append(str(root / "src"))

from rice_ml.unsupervised_learning.k_means_clustering import k_means_clustering


def test_kmeans_converges_to_expected_centers():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=-2, scale=0.1, size=(10, 2)),
        rng.normal(loc=2, scale=0.1, size=(10, 2)),
    ])

    np.random.seed(0)
    model = k_means_clustering(n_clusters=2, metric="euclidean", max_iter=20, tol=1e-6)
    labels = model.fit_predict(X)

    assert set(labels) == {0, 1}
    centers = model.cluster_centers_
    assert np.allclose(np.sort(centers[:, 0]), np.array([-2.0, 2.0]), atol=0.2)
