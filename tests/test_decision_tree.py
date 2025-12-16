import sys
from pathlib import Path

import numpy as np
import pytest

root = Path().resolve()
while not (root / "src" / "rice_ml").exists() and root != root.parent:
    root = root.parent
sys.path.append(str(root / "src"))

from rice_ml.supervised_learning.decision_tree import decision_tree


def test_decision_tree_classifies_simple_threshold():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    clf = decision_tree(max_depth=2, random_state=0)
    clf.fit(X, y)

    preds = clf.predict(np.array([[0.2], [2.5]]))
    assert np.array_equal(preds, np.array([0, 1]))


def test_decision_tree_predict_proba_sums_to_one():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    clf = decision_tree(max_depth=3, random_state=42)
    clf.fit(X, y)

    proba = clf.predict_proba(np.array([[0.5], [2.5]]))
    assert proba.shape == (2, 2)
    assert np.allclose(np.sum(proba, axis=1), np.ones(2))


def test_decision_tree_rejects_negative_impurity_threshold():
    with pytest.raises(ValueError):
        decision_tree(min_impurity_decrease=-0.1)