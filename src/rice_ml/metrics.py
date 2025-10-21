"""
metrics.py
----------

Distance metrics for KNN.

Dependencies
------------
- numpy
"""

from typing import Iterable
import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    L2 (Euclidean) distance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.linalg.norm(x - y))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    L1 (Manhattan) distance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sum(np.abs(x - y)))


def pairwise_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Efficient pairwise Euclidean distances using (a-b)^2 = a^2 + b^2 - 2ab.

    Parameters
    ----------
    A : np.ndarray, shape (n_a, d)
    B : np.ndarray, shape (n_b, d)

    Returns
    -------
    D : np.ndarray, shape (n_a, n_b)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A2 = np.sum(A**2, axis=1, keepdims=True)
    B2 = np.sum(B**2, axis=1, keepdims=True).T
    D2 = A2 + B2 - 2.0 * A @ B.T
    # numerical guard
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2, out=D2)


def pairwise_distances(
    A: np.ndarray,
    B: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """
    Generic pairwise distance using any (x,y)->float metric.
    Falls back to row-wise apply; vectorized only for known metrics.
    """
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)

    # Fast-path for euclidean
    if metric is euclidean_distance:
        return pairwise_euclidean(A, B)

    D = np.empty((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        D[i] = np.apply_along_axis(lambda b: metric(A[i], b), axis=1, arr=B)
    return D


def classification_error(y_true, y_pred):
    """
    Compute the classification error rate.

    The classification error is defined as the proportion of incorrect predictions:
        error = 1 - accuracy = 1 - (number of correct predictions / total samples)

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    float
        Classification error, a value between 0.0 (perfect predictions)
        and 1.0 (all predictions incorrect).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 1 - np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared error (MSE) between true and predicted values.

    The MSE measures the average of the squared differences between predictions
    and actual values:
        MSE = mean((y_true - y_pred)^2)

    Parameters
    ----------
    y_true : array-like
        True continuous target values.
    y_pred : array-like
        Predicted continuous target values.

    Returns
    -------
    float
        Mean squared error. Lower values indicate better predictive accuracy.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)
