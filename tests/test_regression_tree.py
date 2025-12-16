import sys
from pathlib import Path

import numpy as np

root = Path().resolve()
while not (root / "src" / "rice_ml").exists() and root != root.parent:
    root = root.parent
sys.path.append(str(root / "src"))

from rice_ml.supervised_learning.regression_tree import regression_tree


def test_regression_tree_fits_piecewise_constant():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 1.0, 3.0, 3.0])

    reg = regression_tree(max_depth=2, random_state=0)
    reg.fit(X, y)

    preds = reg.predict(np.array([[0.5], [2.5]])).ravel()
    assert np.allclose(preds, np.array([1.0, 3.0]), atol=0.1)


def test_regression_tree_feature_importances_focus_on_informative_feature():
    rng = np.random.default_rng(0)
    X = np.column_stack((rng.normal(0, 1, size=50), rng.normal(0, 1, size=50)))
    y = 2 * X[:, 0] + rng.normal(0, 0.1, size=50)

    reg = regression_tree(max_depth=3, random_state=0)
    reg.fit(X, y)

    importances = reg.feature_importances_
    assert importances is not None
    assert np.isclose(importances.sum(), 1.0)
    assert importances[0] > importances[1]