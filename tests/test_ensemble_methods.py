import sys
from pathlib import Path

import numpy as np
import pytest

root = Path().resolve()
while not (root / "src" / "rice_ml").exists() and root != root.parent:
    root = root.parent
sys.path.append(str(root / "src"))

from rice_ml.supervised_learning.ensemble_methods import bagging_classifier


def test_bagging_classifier_majority_vote():
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])

    clf = bagging_classifier(n_estimators=5, max_samples=0.8, random_state=0)
    clf.fit(X, y)

    preds = clf.predict(np.array([[0.1], [0.9]]))
    assert np.array_equal(preds, np.array([0, 1]))


def test_bagging_classifier_predict_requires_fit():
    clf = bagging_classifier(n_estimators=3)
    with pytest.raises(RuntimeError):
        clf.predict(np.array([[0.0]]))


def test_bagging_classifier_feature_importances_are_meaned():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)

    clf = bagging_classifier(n_estimators=4, random_state=1)
    clf.fit(X, y)

    importances = clf.feature_importances()
    assert importances.shape == (2,)
    assert np.isclose(importances.sum(), 1.0)
    assert importances[0] > importances[1]