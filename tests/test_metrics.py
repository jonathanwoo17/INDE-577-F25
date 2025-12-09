import sys
from pathlib import Path

# Start at the current directory
root = Path().resolve()

while not (root / "src" / "rice_ml").exists() and root != root.parent:
    root = root.parent

sys.path.append(str(root / "src"))

# tests/test_metrics.py
import numpy as np
import pytest

from rice_ml.supervised_learning.metrics import (
    euclidean_distance,
    manhattan_distance,
    pairwise_euclidean,
    pairwise_distances,
    classification_error,
    mean_squared_error,
)


# ------------------------- Distance primitives -------------------------

def test_euclidean_distance_basic():
    # 1D
    assert euclidean_distance(np.array([0.0]), np.array([0.0])) == 0.0
    assert np.isclose(euclidean_distance(np.array([1.0]), np.array([4.0])), 3.0)

    # 2D
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert np.isclose(euclidean_distance(a, b), 5.0)

    # symmetry
    c = np.array([-1.2, 7.3, 0.5])
    d = np.array([ 2.0, 1.3, 0.5])
    assert np.isclose(euclidean_distance(c, d), euclidean_distance(d, c))


def test_manhattan_distance_basic():
    # 1D
    assert manhattan_distance(np.array([0.0]), np.array([0.0])) == 0.0
    assert manhattan_distance(np.array([1.0]), np.array([4.0])) == 3.0

    # 2D
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert manhattan_distance(a, b) == 7.0

    # symmetry
    c = np.array([-1.2, 7.3, 0.5])
    d = np.array([ 2.0, 1.3, 0.5])
    assert manhattan_distance(c, d) == manhattan_distance(d, c)


def test_triangle_inequality_spot_checks():
    # Euclidean triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    c = np.array([3.0, 1.0])
    d_ac = euclidean_distance(a, c)
    d_ab = euclidean_distance(a, b)
    d_bc = euclidean_distance(b, c)
    assert d_ac <= d_ab + d_bc + 1e-12

    # Manhattan triangle inequality
    d_ac_L1 = manhattan_distance(a, c)
    d_ab_L1 = manhattan_distance(a, b)
    d_bc_L1 = manhattan_distance(b, c)
    assert d_ac_L1 <= d_ab_L1 + d_bc_L1 + 1e-12


# ------------------------- Pairwise Euclidean -------------------------

def test_pairwise_euclidean_matches_naive():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(7, 3))
    B = rng.normal(size=(5, 3))

    # vectorized
    D_vec = pairwise_euclidean(A, B)

    # naive
    D_naive = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            D_naive[i, j] = euclidean_distance(A[i], B[j])

    assert D_vec.shape == (7, 5)
    assert np.allclose(D_vec, D_naive, atol=1e-12)
    assert np.all(D_vec >= -1e-12)  # numerical guard non-negativity


def test_pairwise_euclidean_self_distances_zero():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 4))
    D = pairwise_euclidean(A, A)
    # diagonal should be ~0 and symmetric
    assert np.allclose(np.diag(D), 0.0, atol=1e-12)
    assert np.allclose(D, D.T, atol=1e-12)
    assert np.all(D >= -1e-12)


# ------------------------- Generic pairwise_distances -------------------------

def test_pairwise_distances_euclidean_fastpath_equals_pairwise_euclidean():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(8, 3))
    B = rng.normal(size=(9, 3))

    D_generic = pairwise_distances(A, B, metric=euclidean_distance)
    D_fast = pairwise_euclidean(A, B)

    assert D_generic.shape == (8, 9)
    assert np.allclose(D_generic, D_fast, atol=1e-12)


def test_pairwise_distances_with_manhattan_matches_naive():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(5, 2))
    B = rng.normal(size=(6, 2))

    D_generic = pairwise_distances(A, B, metric=manhattan_distance)

    # naive L1
    D_naive = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            D_naive[i, j] = manhattan_distance(A[i], B[j])

    assert np.allclose(D_generic, D_naive, atol=1e-12)


def test_pairwise_distances_maintains_nonnegativity():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(4, 5))
    B = rng.normal(size=(7, 5))
    D = pairwise_distances(A, B, metric=euclidean_distance)
    assert np.all(D >= -1e-12)


# ------------------------- Classification error -------------------------

def test_classification_error_perfect_and_all_wrong():
    y_true = np.array([0, 1, 1, 0, 2, 2])
    y_pred_perfect = np.array([0, 1, 1, 0, 2, 2])
    y_pred_wrong =   np.array([1, 0, 0, 2, 1, 0])

    assert classification_error(y_true, y_pred_perfect) == 0.0
    # 0/6 correct -> error 1.0
    assert classification_error(y_true, y_pred_wrong) == 1.0


def test_classification_error_mixed_and_string_labels():
    y_true = np.array(["cat", "dog", "dog", "cat", "dog"])
    y_pred = np.array(["cat", "cat", "dog", "cat", "dog"])
    # correct: positions 0,2,3,4 => 4/5 -> accuracy 0.8 -> error 0.2
    assert np.isclose(classification_error(y_true, y_pred), 0.2)


def test_classification_error_empty_returns_nan():
    y_true = np.array([])
    y_pred = np.array([])
    err = classification_error(y_true, y_pred)
    assert np.isnan(err)


# ------------------------- Mean squared error -------------------------

def test_mean_squared_error_perfect_and_constant_offset():
    y_true = np.array([0.0, 1.0, -2.0, 3.0])
    y_pred_perfect = y_true.copy()
    assert mean_squared_error(y_true, y_pred_perfect) == 0.0

    # constant offset c -> MSE == c^2
    c = 0.75
    y_pred_offset = y_true + c
    assert np.isclose(mean_squared_error(y_true, y_pred_offset), c**2)


def test_mean_squared_error_types_and_empty():
    y_true = np.array([1, 2, 3, 4], dtype=int)
    y_pred = np.array([1.0, 2.0, 2.5, 3.0], dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    assert isinstance(mse, float)
    assert mse >= 0.0

    # Empty -> numpy mean of empty slice -> NaN: document current behavior
    assert np.isnan(mean_squared_error(np.array([]), np.array([])))
