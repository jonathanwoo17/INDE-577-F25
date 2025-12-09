import sys
from pathlib import Path

# Start at the current directory
root = Path().resolve()

while not (root / "src" / "rice_ml").exists() and root != root.parent:
    root = root.parent

sys.path.append(str(root / "src"))

# tests/test_preprocess.py
import numpy as np
import pytest

from rice_ml.supervised_learning import preprocess as pp


# ---------------------------
# Helpers
# ---------------------------

def make_Xy(n=10, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 2, size=n)
    return X, y


# ---------------------------
# train_test_split
# ---------------------------

def test_train_test_split_shapes_and_types_default():
    X, y = make_Xy(n=50, d=5)
    Xtr, Xte, ytr, yte = pp.train_test_split(X, y)  # test_size=0.2 -> 10 test
    assert Xtr.shape == (40, 5)
    assert Xte.shape == (10, 5)
    assert ytr.shape == (40,)
    assert yte.shape == (10,)
    # Ensure returns are numpy arrays (not views with unexpected types)
    assert isinstance(Xtr, np.ndarray) and isinstance(Xte, np.ndarray)
    assert isinstance(ytr, np.ndarray) and isinstance(yte, np.ndarray)


def test_train_test_split_raises_on_mismatched_rows():
    X = np.zeros((5, 2))
    y = np.zeros(6)
    with pytest.raises(ValueError, match="same number of rows"):
        pp.train_test_split(X, y)


@pytest.mark.parametrize(
    "n,test_size,expected_test",
    [
        (10, 0.0, 0),
        (10, 1.0, 10),
        (5, 0.4, 2),   # int(round(2.0)) = 2
        (5, 0.5, 2),   # np.round uses bankers' rounding: round(2.5) -> 2
        (5, 0.6, 3),
        (7, 0.2857, int(np.round(7 * 0.2857))),
    ],
)
def test_train_test_split_rounding_and_edges(n, test_size, expected_test):
    X, y = make_Xy(n=n, d=3)
    Xtr, Xte, ytr, yte = pp.train_test_split(X, y, test_size=test_size, shuffle=False)
    assert Xte.shape[0] == expected_test
    assert Xtr.shape[0] == n - expected_test
    # shuffle=False => test set should be first n_test rows
    np.testing.assert_allclose(Xte, X[:expected_test])
    np.testing.assert_allclose(yte, y[:expected_test])
    np.testing.assert_allclose(Xtr, X[expected_test:])
    np.testing.assert_allclose(ytr, y[expected_test:])


def test_train_test_split_deterministic_with_random_state():
    X, y = make_Xy(n=30, d=4)
    out1 = pp.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=123)
    out2 = pp.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=123)
    # all four arrays should match exactly
    for a, b in zip(out1, out2):
        np.testing.assert_array_equal(a, b)


def test_train_test_split_randomization_changes_with_different_seed():
    X, y = make_Xy(n=30, d=4)
    out1 = pp.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)
    out2 = pp.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=2)
    # at least one of the splits should differ
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(out1[0], out2[0])  # X_train differs likely


# ---------------------------
# standardize_fit & standardize_transform
# ---------------------------

def test_standardize_fit_basic_and_zero_variance_handling():
    X = np.array([[1.0, 5.0, 2.0],
                  [3.0, 5.0, 6.0],
                  [5.0, 5.0, 10.0]])  # middle column has zero variance
    mean, std = pp.standardize_fit(X)
    np.testing.assert_allclose(mean, [3.0, 5.0, 6.0])
    # std computed with ddof=0; zero-variance column should be replaced by 1.0
    assert std[1] == 1.0
    # other std values
    # For col0: std population of [1,3,5] is sqrt(((4+0+4)/3)) = sqrt(8/3)
    # For col2: [2,6,10] -> same pattern -> sqrt(8/3)
    np.testing.assert_allclose(std[[0, 2]], [np.sqrt(8/3), np.sqrt(32/3)])


def test_standardize_transform_produces_zero_mean_unit_var_when_possible():
    X, _ = make_Xy(n=100, d=4)
    mean, std = pp.standardize_fit(X)
    Z = pp.standardize_transform(X, mean, std)
    # Columns with non-zero original std should have ~0 mean and ~1 std
    np.testing.assert_allclose(Z.mean(axis=0), np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0, ddof=0), np.ones(4), atol=1e-12)


def test_standardize_transform_round_trip_reconstruction():
    X = np.array([[2., 2., 2.],
                  [4., 2., 6.],
                  [6., 2., 10.]])  # col1 constant
    mean, std = pp.standardize_fit(X)
    Z = pp.standardize_transform(X, mean, std)
    X_rec = Z * std + mean
    np.testing.assert_allclose(X_rec, X, atol=1e-12)


def test_standardize_transform_broadcasting_and_dtype():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
    mean, std = pp.standardize_fit(X)
    Z = pp.standardize_transform(X, mean, std)
    assert Z.dtype == float
    # broadcasting should work with 1D mean/std
    assert mean.shape == (2,) and std.shape == (2,)


def test_standardize_does_not_modify_inputs():
    X0 = np.array([[1., 2.], [3., 4.]], dtype=float)
    X = X0.copy()
    mean, std = pp.standardize_fit(X)
    Z = pp.standardize_transform(X, mean, std)
    # inputs unchanged
    np.testing.assert_array_equal(X, X0)
    # output is a different object
    assert not np.shares_memory(Z, X)


# ---------------------------
# minmax_fit & minmax_transform
# ---------------------------

def test_minmax_fit_basic_and_zero_range_handling():
    X = np.array([[0.0, 5.0, 2.0],
                  [2.0, 5.0, 6.0],
                  [4.0, 5.0, 10.0]])  # middle column constant
    mn, rg = pp.minmax_fit(X)
    np.testing.assert_allclose(mn, [0.0, 5.0, 2.0])
    np.testing.assert_allclose(rg, [4.0, 1.0, 8.0])  # zero range replaced by 1.0


def test_minmax_transform_scales_to_0_1_range():
    X, _ = make_Xy(n=50, d=3)
    mn, rg = pp.minmax_fit(X)
    S = pp.minmax_transform(X, mn, rg)
    # S should be in [0, 1] up to numerical noise
    assert np.all(S >= -1e-12)
    assert np.all(S <= 1 + 1e-12)


def test_minmax_transform_round_trip_reconstruction():
    X = np.array([[2., 2., 2.],
                  [4., 2., 6.],
                  [6., 2., 10.]])  # col1 constant
    mn, rg = pp.minmax_fit(X)
    S = pp.minmax_transform(X, mn, rg)
    X_rec = S * rg + mn
    np.testing.assert_allclose(X_rec, X, atol=1e-12)


def test_minmax_transform_broadcasting_and_dtype():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
    mn, rg = pp.minmax_fit(X)
    S = pp.minmax_transform(X, mn, rg)
    assert S.dtype == float
    assert mn.shape == (2,) and rg.shape == (2,)


def test_minmax_does_not_modify_inputs():
    X0 = np.array([[1., 2.], [3., 4.]], dtype=float)
    X = X0.copy()
    mn, rg = pp.minmax_fit(X)
    S = pp.minmax_transform(X, mn, rg)
    np.testing.assert_array_equal(X, X0)
    assert not np.shares_memory(S, X)


# ---------------------------
# Additional robustness cases
# ---------------------------

def test_handles_non_contiguous_views():
    X, y = make_Xy(n=20, d=5)
    X_view = X[:, ::-1]  # non-contiguous view
    # fit/transform should still work
    mean, std = pp.standardize_fit(X_view)
    Z = pp.standardize_transform(X_view, mean, std)
    mn, rg = pp.minmax_fit(X_view)
    S = pp.minmax_transform(X_view, mn, rg)
    assert Z.shape == X_view.shape
    assert S.shape == X_view.shape


def test_single_sample_and_single_feature_edges():
    X = np.array([[3.14]])  # 1x1
    y = np.array([1])
    Xtr, Xte, ytr, yte = pp.train_test_split(X, y, test_size=1.0, shuffle=False)
    np.testing.assert_allclose(Xte, X)
    np.testing.assert_allclose(yte, y)
    assert Xtr.size == 0 and ytr.size == 0

    mean, std = pp.standardize_fit(X)
    Z = pp.standardize_transform(X, mean, std)
    np.testing.assert_allclose(Z, [[0.0]])  # zero-variance -> std=1, mean=X => z=0

    mn, rg = pp.minmax_fit(X)
    S = pp.minmax_transform(X, mn, rg)
    np.testing.assert_allclose(S, [[0.0]])  # zero-range -> range=1, so scaled 0
