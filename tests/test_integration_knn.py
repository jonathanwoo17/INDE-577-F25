# tests/test_integration_knn.py
import numpy as np
import pytest

from rice_ml.knn import KNN
from rice_ml import metrics as met
from rice_ml import preprocess as pre
from rice_ml import postprocess as post


# ---------------------------
# Fixtures / small helpers
# ---------------------------

def make_toy_classification(seed=0):
    """
    Two-cluster 2D dataset:
    - Class 0 around (0,0)
    - Class 1 around (3,3)
    """
    rng = np.random.default_rng(seed)
    n = 40
    X0 = rng.normal(loc=0.0, scale=0.3, size=(n//2, 2))
    X1 = rng.normal(loc=3.0, scale=0.3, size=(n//2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y


def make_toy_regression(seed=1):
    """
    1D regression: y = 2x + 1 + noise
    """
    rng = np.random.default_rng(seed)
    n = 60
    X = rng.uniform(-2, 2, size=(n, 1))
    y = 2.0 * X[:, 0] + 1.0 + rng.normal(0, 0.2, size=n)
    return X, y


# ---------------------------
# End-to-end classification
# ---------------------------

@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("distance", ["euclidean", "manhattan"])
def test_classification_pipeline_split_scale_predict(distance, weights):
    X, y = make_toy_classification(seed=42)

    # Split (deterministic)
    Xtr, Xte, ytr, yte = pre.train_test_split(X, y, test_size=0.25, shuffle=True, random_state=7)

    # Standardize using only train stats; apply to both sets
    mean, std = pre.standardize_fit(Xtr)
    Xtr_z = pre.standardize_transform(Xtr, mean, std)
    Xte_z = pre.standardize_transform(Xte, mean, std)

    # Fit KNN
    clf = KNN(k=5, task="classification", distance=distance, weights=weights)
    clf.fit(Xtr_z, ytr)

    # Predict + evaluate
    yhat = clf.predict(Xte_z)
    err = met.classification_error(yte, yhat)

    # Clusters are well separated; error should be very low
    assert err <= 0.10


def test_classification_tie_break_smallest_label_deterministic():
    # Construct neighbors with equal total weight for labels {0,1}
    # Use KNN with return_neighbors, then force equal weights in post step
    X = np.array([[0.0], [0.0], [1.0], [1.0]])  # training features
    y = np.array([0, 1, 0, 1])                  # equal counts of 0 and 1
    q = np.array([[0.5]])                       # equidistant to both groups

    knn = KNN(k=2, task="classification", distance="euclidean", weights="uniform", return_neighbors=True)
    knn.fit(X, y)
    preds, (idxs, dists) = knn.predict(q)

    # With uniform weights and symmetric neighbors, total weights tie; smallest label (0) should win.
    assert preds.shape == (1,)
    assert preds[0] == 0


def test_classification_return_neighbors_sorted_and_finite_with_zero_distance():
    # Include exact duplicate point to test eps handling for distance weights
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 0, 1])
    q = np.array([[0.0, 0.0]])  # exactly equals first training point

    clf = KNN(k=3, task="classification", distance="euclidean", weights="distance", return_neighbors=True)
    clf.fit(X, y)
    preds, (idxs, dists) = clf.predict(q)

    # Neighbors/Distances shapes
    assert idxs.shape == (1, 3)
    assert dists.shape == (1, 3)

    # Distances must be sorted ascending and finite (no inf thanks to eps)
    assert np.all(np.isfinite(dists))
    assert np.all(dists[0] == np.sort(dists[0]))
    # First distance should be ~0 because of the duplicate
    assert dists[0, 0] <= 1e-12
    # Prediction should be a valid class
    assert preds[0] in (0, 1)


# ---------------------------
# End-to-end regression
# ---------------------------

@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("distance", ["euclidean", "manhattan"])
def test_regression_pipeline_split_scale_predict(distance, weights):
    X, y = make_toy_regression(seed=5)

    # Split (deterministic)
    Xtr, Xte, ytr, yte = pre.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=99)

    # Min-max scale using train only
    mn, rg = pre.minmax_fit(Xtr)
    Xtr_s = pre.minmax_transform(Xtr, mn, rg)
    Xte_s = pre.minmax_transform(Xte, mn, rg)

    # Fit/predict
    reg = KNN(k=7, task="regression", distance=distance, weights=weights)
    reg.fit(Xtr_s, ytr)
    yhat = reg.predict(Xte_s)

    # Evaluate MSE — should be clearly better than naive mean baseline
    mse = met.mean_squared_error(yte, yhat)
    mse_baseline = met.mean_squared_error(yte, np.full_like(yte, fill_value=ytr.mean(), dtype=float))
    assert mse < mse_baseline


def test_regression_average_label_matches_manual_weighted_mean():
    # Manually craft 1 query where weights should drive prediction
    Xtr = np.array([[0.0], [1.0], [2.0]])
    ytr = np.array([0.0, 10.0, 20.0])
    Xq  = np.array([[1.0]])

    # distance weighting => center point dominates
    reg = KNN(k=3, task="regression", distance="euclidean", weights="distance")
    reg.fit(Xtr, ytr)
    yhat = reg.predict(Xq)

    # Manual compute weights
    dists = np.array([1.0, 0.0, 1.0])
    w = 1.0 / (dists + reg.eps)
    expected = float(np.dot(w, ytr) / w.sum())
    assert np.allclose(yhat[0], expected, rtol=1e-12, atol=1e-12)


# ---------------------------
# kneighbors / distance registry / callable metric
# ---------------------------

def test_kneighbors_sorted_indices_and_distances_match_metric():
    X, y = make_toy_classification(seed=10)
    q = X[:3]  # three queries from the training set

    knn = KNN(k=4, task="classification", distance="euclidean", weights="uniform", return_neighbors=True)
    knn.fit(X, y)
    _, (idxs, dists) = knn.predict(q)

    # Distances should match pairwise computation row-wise
    Dfull = met.pairwise_distances(q, X, metric=met.euclidean_distance)
    D4 = np.take_along_axis(Dfull, idxs, axis=1)
    assert np.allclose(dists, D4, rtol=1e-12, atol=1e-12)
    # Sorted by distance per row
    assert np.all(dists == np.sort(dists, axis=1))


def test_callable_metric_is_used_and_ordering_respected():
    # Custom metric = scaled Euclidean (monotone; ordering should be identical)
    def scaled_euclidean(a: np.ndarray, b: np.ndarray) -> float:
        return 3.0 * met.euclidean_distance(a, b)

    X, y = make_toy_classification(seed=123)
    q = X[:5]

    knn = KNN(k=3, task="classification", distance=scaled_euclidean, weights="uniform", return_neighbors=True)
    knn.fit(X, y)
    _, (idxs_scaled, dists_scaled) = knn.predict(q)

    # Compare to plain Euclidean
    knn2 = KNN(k=3, task="classification", distance="euclidean", weights="uniform", return_neighbors=True)
    knn2.fit(X, y)
    _, (idxs_euc, dists_euc) = knn2.predict(q)

    # Indices should match (same ordering); distances scaled by ~3
    assert np.array_equal(idxs_scaled, idxs_euc)
    assert np.allclose(dists_scaled, 3.0 * dists_euc, rtol=1e-12, atol=1e-12)


# ---------------------------
# Single-sample shapes / 1D query handling
# ---------------------------

def test_single_query_and_1d_input_are_handled():
    X, y = make_toy_classification(seed=7)
    knn = KNN(k=3, task="classification", distance="euclidean", weights="uniform", return_neighbors=True).fit(X, y)

    # 1D query reshaped internally
    q = X[0]           # shape (2,)
    preds, (idxs, dists) = knn.predict(q)

    assert preds.shape == (1,)
    assert idxs.shape == (1, 3)
    assert dists.shape == (1, 3)


# ---------------------------
# Integration with preprocess: round-trip scale + split determinism
# ---------------------------

def test_split_determinism_and_scaling_round_trip_on_train():
    X, y = make_toy_classification(seed=2024)
    Xtr, Xte, ytr, yte = pre.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=777)

    # Standardize on Xtr
    mu, sd = pre.standardize_fit(Xtr)
    Ztr = pre.standardize_transform(Xtr, mu, sd)
    Xtr_rec = Ztr * sd + mu
    np.testing.assert_allclose(Xtr_rec, Xtr, atol=1e-12)

    # Split determinism (same seed -> identical)
    Xtr2, Xte2, ytr2, yte2 = pre.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=777)
    for a, b in zip((Xtr, Xte, ytr, yte), (Xtr2, Xte2, ytr2, yte2)):
        np.testing.assert_array_equal(a, b)


# ---------------------------
# Failure paths that cross boundaries
# ---------------------------

def test_fit_then_kneighbors_guard_violations():
    X, y = make_toy_classification(seed=11)
    knn = KNN(k=5, task="classification").fit(X, y)

    with pytest.raises(ValueError, match="positive"):
        knn.kneighbors(X[:2], n_neighbors=0)

    with pytest.raises(ValueError, match="exceed"):
        knn.kneighbors(X[:2], n_neighbors=X.shape[0] + 1)


def test_predict_requires_fit_and_handles_k_gt_train_samples():
    X, y = make_toy_classification(seed=9)
    knn = KNN(k=5, task="classification")

    with pytest.raises(RuntimeError, match="not fitted"):
        knn.predict(X[:2])

    with pytest.raises(ValueError, match="exceed"):
        KNN(k=X.shape[0] + 1, task="classification").fit(X, y)
