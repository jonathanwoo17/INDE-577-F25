import sys
from pathlib import Path

import numpy as np
import pytest

root = Path().resolve()
while not (root / "src" / "rice_ml").exists() and root != root.parent:
    root = root.parent
sys.path.append(str(root / "src"))

from rice_ml.unsupervised_learning.dbscan import DBScan


def test_dbscan_two_clusters_with_noise():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [10.0, 10.0],  # noise far from both clusters
        ]
    )

    model = DBScan(eps=0.2, min_samples=2)
    labels, n_clusters = model.fit_predict(X)

    assert n_clusters == 2
    assert set(labels) == {0, 1, 2}
    assert labels[-1] == 0


def test_dbscan_rejects_unknown_metric():
    with pytest.raises(ValueError):
        DBScan(metric="chebyshev")


def test_dbscan_manhattan_distance_clusters_linear_points():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
        ]
    )

    model = DBScan(eps=1.1, min_samples=2, metric="manhattan")
    labels, n_clusters = model.fit_predict(X)

    assert n_clusters == 2
    assert set(labels) == {1, 2}