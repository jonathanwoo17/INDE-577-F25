import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_kmeans_predict_requires_fit():
    model = k_means_clustering(n_clusters=2)
    with pytest.raises(RuntimeError):
        model.predict(np.array([[0.0, 0.0]]))


def test_kmeans_supports_manhattan_metric():
    np.random.seed(1)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])

    model = k_means_clustering(n_clusters=2, metric="manhattan", max_iter=10, tol=1e-6)
    labels = model.fit_predict(X)

    assert set(labels) == {0, 1}
    centers = np.sort(model.cluster_centers_[:, 0])
    assert np.allclose(centers, np.array([0.0, 5.0]), atol=0.1)