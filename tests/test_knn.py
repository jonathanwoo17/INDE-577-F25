# tests/test_knn.py
import numpy as np
import pytest

from rice_ml.supervised_learning.knn import KNN


# ------------------------- Fixtures (self-contained) -------------------------

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def sep_binary(rng):
    """Two clearly separable 2D blobs (binary classification)."""
    X0 = rng.normal(loc=[-2.5, -2.0], scale=0.25, size=(30, 2))
    X1 = rng.normal(loc=[+2.5, +2.0], scale=0.25, size=(30, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1))
    return X, y


@pytest.fixture
def sep_multiclass(rng):
    """Three well-separated 2D blobs (multiclass labels 0/1/2)."""
    centers = np.array([[-2.0, -2.0], [2.2, 0.0], [0.0, 2.5]])
    X_parts, y_parts = [], []
    for k, c in enumerate(centers):
        X_parts.append(rng.normal(loc=c, scale=0.35, size=(25, 2)))
        y_parts.append(np.full(25, k))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


@pytest.fixture
def manhattan_vs_euclid_case():
    """
    Geometry where L1 and L2 nearest neighbors can disagree for the query at origin.
    """
    X = np.array(
        [
            [1.0, 1.0],   # L2=√2, L1=2  (class 0)
            [2.0, 0.0],   # L2=2,  L1=2  (class 1)
            [0.0, 2.0],   # L2=2,  L1=2  (class 1)
            [1.5, 0.2],   # ~L2=1.513, L1=1.7 (class 1)
        ],
        dtype=float,
    )
    y = np.array([0, 1, 1, 1])
    q = np.array([[0.0, 0.0]])
    return X, y, q


@pytest.fixture
def regression_1d(rng):
    """Simple 1-D regression: y = 2x + 1 + noise."""
    X = rng.uniform(-2, 2, size=(80, 1))
    y = 2.0 * X[:, 0] + 1.0 + rng.normal(0, 0.05, size=80)
    return X, y


# ------------------------- Constructor & validation -------------------------

def test_init_validates_arguments():
    with pytest.raises(ValueError):
        KNN(k=0)
    with pytest.raises(ValueError):
        KNN(k=-2)
    with pytest.raises(ValueError):
        KNN(task="clsfy")  # invalid
    with pytest.raises(ValueError):
        KNN(distance="chebyshev")  # not in registry
    with pytest.raises(ValueError):
        KNN(weights="funky")


def test_fit_validates_shapes_and_k(sep_binary):
    X, y = sep_binary
    model = KNN(k=3, task="classification")
    out = model.fit(X, y)
    assert out is model
    assert model.X_ is not None and model.y_ is not None
    assert model.X_.shape == X.shape
    assert model.y_.shape == y.shape

    with pytest.raises(ValueError):
        KNN().fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))  # X must be 2D
    with pytest.raises(ValueError):
        KNN().fit(np.zeros((3, 2)), np.array([[0], [1], [0]]))    # y must be 1D
    with pytest.raises(ValueError):
        KNN().fit(np.zeros((3, 2)), np.array([0, 1]))             # length mismatch
    with pytest.raises(ValueError):
        KNN(k=len(X) + 1).fit(X, y)                                # k > n_train


def test_predict_requires_fit():
    model = KNN()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((2, 2)))


# ------------------------- kneighbors behavior -------------------------

def test_kneighbors_shapes_sorted_and_custom_k(sep_binary):
    X, y = sep_binary
    model = KNN(k=5).fit(X, y)

    idxs, dists = model.kneighbors(X[:7], return_distance=True)
    assert idxs.shape == (7, 5)
    assert dists.shape == (7, 5)
    # distances sorted ascending per row
    assert np.all(np.diff(dists, axis=1) >= -1e-12)

    idxs_only = model.kneighbors(X[:3], n_neighbors=3, return_distance=False)
    assert idxs_only.shape == (3, 3)

    with pytest.raises(ValueError):
        model.kneighbors(X[:1], n_neighbors=0)
    with pytest.raises(ValueError):
        model.kneighbors(X[:1], n_neighbors=len(X) + 1)

    # 1-D query must be reshaped internally
    idxs_1d, dists_1d = model.kneighbors(X[0], return_distance=True)
    assert idxs_1d.shape == (1, 5)
    assert dists_1d.shape == (1, 5)


def test_manhattan_and_euclidean_can_disagree(manhattan_vs_euclid_case):
    X, y, q = manhattan_vs_euclid_case
    m_e = KNN(k=1, task="classification", distance="euclidean").fit(X, y)
    m_m = KNN(k=1, task="classification", distance="manhattan").fit(X, y)
    pe = m_e.predict(q)[0]
    pm = m_m.predict(q)[0]
    assert pe != pm  # constructed for possible disagreement


# ------------------------- Classification: predict -------------------------

@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_classification_high_train_accuracy(sep_binary, metric, weights):
    X, y = sep_binary
    model = KNN(k=3, task="classification", distance=metric, weights=weights).fit(X, y)
    yhat = model.predict(X)
    assert yhat.shape == y.shape
    assert (yhat == y).mean() >= 0.95  # separable blobs


def test_classification_handles_1d_query(sep_binary):
    X, y = sep_binary
    model = KNN(k=3, task="classification").fit(X, y)
    q = X[0]  # shape (n_features,)
    pred = model.predict(q)
    assert pred.shape == (1,)


def test_classification_return_neighbors_tuple(sep_binary):
    X, y = sep_binary
    model = KNN(k=5, task="classification", return_neighbors=True).fit(X, y)
    preds, (idxs, dists) = model.predict(X[:4])
    assert preds.shape == (4,)
    assert idxs.shape == (4, 5)
    assert dists.shape == (4, 5)
    assert np.all(np.diff(dists, axis=1) >= -1e-12)
    # Neighbor labels correspond to y at those indices (sanity)
    assert np.array_equal(y[idxs][0], y[idxs[0]])


def test_classification_exact_match_with_distance_weights(sep_binary):
    """
    If a query exactly matches a training point, distance weights (1/(d+eps))
    should heavily favor that point -> predicted label equals its label.
    """
    X, y = sep_binary
    model = KNN(k=5, task="classification", weights="distance").fit(X, y)
    q = X[10].copy()  # exact training point
    pred = model.predict(q)[0]
    assert pred == y[10]


# ------------------------- Regression: predict -------------------------

def test_regression_uniform_mse_on_train(regression_1d):
    X, y = regression_1d
    model = KNN(k=5, task="regression", distance="euclidean", weights="uniform").fit(X, y)
    yhat = model.predict(X)
    assert yhat.shape == y.shape
    mse = np.mean((yhat - y) ** 2)
    assert mse < 0.05  # KNN smoothing of near-linear function


def test_regression_distance_vs_uniform_closer_to_nearest(regression_1d):
    """
    For a query near a particular training point, distance weights should
    pull prediction closer to that neighbor's label than uniform weights.
    """
    X, y = regression_1d
    mu = KNN(k=5, task="regression", weights="uniform").fit(X, y)
    md = KNN(k=5, task="regression", weights="distance").fit(X, y)

    q = np.array([[X[0, 0] + 1e-6]])
    yu = mu.predict(q)[0]
    yd = md.predict(q)[0]
    y_near = y[0]
    assert abs(yd - y_near) <= abs(yu - y_near) + 1e-12


def test_regression_handles_1d_query(regression_1d):
    X, y = regression_1d
    model = KNN(k=3, task="regression").fit(X, y)
    pred = model.predict(X[1])  # 1-D query
    assert pred.shape == (1,)


def test_regression_return_neighbors_tuple(regression_1d):
    X, y = regression_1d
    model = KNN(k=4, task="regression", return_neighbors=True).fit(X, y)
    preds, (idxs, dists) = model.predict(X[:6])
    assert preds.shape == (6,)
    assert idxs.shape == (6, 4)
    assert dists.shape == (6, 4)
    assert np.all(np.diff(dists, axis=1) >= -1e-12)


# ------------------------- Mixed / edge cases -------------------------

def test_custom_callable_distance(sep_binary):
    """Passing a callable for `distance` should work and be used."""
    def scaled_l2(a, b):
        return float(np.linalg.norm(a - b) * 0.25)

    X, y = sep_binary
    model = KNN(k=3, task="classification", distance=scaled_l2).fit(X, y)
    yhat = model.predict(X[:7])
    assert yhat.shape == (7,)


def test_predict_does_not_mutate_training_arrays(sep_binary):
    X, y = sep_binary
    Xc, yc = X.copy(), y.copy()
    model = KNN(k=3).fit(X, y)
    _ = model.predict(X[:5])
    assert np.array_equal(X, Xc)
    assert np.array_equal(y, yc)


def test_euclidean_scaling_invariance(sep_binary):
    """
    Scaling both train and query by a positive constant should not change the
    nearest-neighbor decision for Euclidean metric (up to rare tie flips).
    """
    X, y = sep_binary
    scale = 7.75
    m1 = KNN(k=3, task="classification", distance="euclidean").fit(X, y)
    m2 = KNN(k=3, task="classification", distance="euclidean").fit(X * scale, y)
    y1 = m1.predict(X)
    y2 = m2.predict(X * scale)
    assert (y1 == y2).mean() > 0.98


def test_predict_output_dtype_and_shape(sep_binary):
    X, y = sep_binary
    model = KNN(k=3, task="classification").fit(X, y)
    preds = model.predict(X[:9])
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (9,)


def test_kneighbors_respects_default_k_when_none(sep_binary):
    X, y = sep_binary
    model = KNN(k=4).fit(X, y)
    idxs, dists = model.kneighbors(X[:2], n_neighbors=None, return_distance=True)
    assert idxs.shape == (2, 4)
    assert dists.shape == (2, 4)


def test_zero_distance_results_in_finite_weights_and_prediction(sep_binary):
    """
    Create duplicate rows so zero distances appear; ensure prediction is valid
    (no NaNs/Infs), and neighbors include a zero distance.
    """
    X, y = sep_binary
    # Duplicate a point
    X = np.vstack([X, X[0:1]])
    y = np.concatenate([y, y[0:1]])
    model = KNN(k=5, task="classification", weights="distance", return_neighbors=True).fit(X, y)
    preds, (idxs, dists) = model.predict(X[0])
    assert preds.shape == (1,)
    assert np.isfinite(preds).all()
    assert np.any(np.isclose(dists, 0.0))

def test_k_equals_n_samples_ok(sep_binary):
    """k == n_train should be allowed and produce valid predictions."""
    X, y = sep_binary
    k_all = len(X)
    model = KNN(k=k_all, task="classification").fit(X, y)
    preds = model.predict(X[:3])
    assert preds.shape == (3,)


def test_non_numeric_labels_classification():
    """String labels should be supported for classification."""
    X = np.array([[0.0, 0.0],
                  [1.0, 1.0],
                  [0.9, 1.1],
                  [-1.0, -1.0]])
    y = np.array(["cat", "dog", "dog", "cat"], dtype=object)
    model = KNN(k=3, task="classification", distance="euclidean").fit(X, y)
    preds = model.predict(np.array([[0.95, 1.05], [-0.5, -0.6]]))
    assert preds.shape == (2,)
    assert set(preds.tolist()).issubset({"cat", "dog"})


def test_regression_int_labels_returns_float():
    """Even if y is integer-typed, regression predictions should be float."""
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = (2.0 * X[:, 0] + 1.0).astype(int)  # ints on purpose
    model = KNN(k=3, task="regression", distance="euclidean").fit(X, y)
    pred = model.predict(np.array([[0.5]]))[0]
    assert isinstance(pred, float)


def test_predict_empty_batch_returns_empty_array(sep_binary):
    """Predicting on an empty query batch should return an empty 1D array."""
    X, y = sep_binary
    model = KNN(k=3, task="classification").fit(X, y)
    empty_queries = np.empty((0, X.shape[1]), dtype=float)
    preds = model.predict(empty_queries)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (0,)


def test_kneighbors_return_indices_only_type(sep_binary):
    """kneighbors(..., return_distance=False) returns indices only."""
    X, y = sep_binary
    model = KNN(k=4).fit(X, y)
    idxs = model.kneighbors(X[:2], return_distance=False)
    assert isinstance(idxs, np.ndarray)
    assert idxs.shape == (2, 4)


def test_predict_raises_on_feature_mismatch(sep_binary):
    """Predict with wrong feature dimension should error cleanly."""
    X, y = sep_binary  # shape (n, 2)
    model = KNN(k=3).fit(X, y)
    with pytest.raises(Exception):
        model.predict(np.array([[0.1, 0.2, 0.3]]))  # 3 features != 2
