import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

root = Path().resolve()
while not (root / "src" / "rice_ml").exists() and root != root.parent:
    root = root.parent
sys.path.append(str(root / "src"))

from rice_ml.unsupervised_learning.pca import pca


def test_pca_reduces_dimension_and_variance_ordering():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)

    model = pca(n_components=1, scale=False)
    transformed = model.fit_transform(X)

    assert transformed.shape == (4, 1)
    assert model.explained_variance_ratio_[0] > 0.9


def test_pca_transform_requires_fit():
    model = pca(n_components=1, scale=True)
    with pytest.raises(RuntimeError):
        model.transform(np.array([[1.0, 2.0]]))


def test_pca_scaling_tracks_feature_mean():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

    model = pca(n_components=2, scale=True)
    transformed = model.fit_transform(X)

    assert transformed.shape[1] == 2
    assert np.allclose(model.mean_, np.array([3.0, 4.0]))
    assert np.isclose(np.sum(model.explained_variance_ratio_), 1.0)