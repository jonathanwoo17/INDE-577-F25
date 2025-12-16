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

