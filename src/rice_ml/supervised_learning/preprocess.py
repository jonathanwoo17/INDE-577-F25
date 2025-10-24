"""
preprocess.py
-------------

Utility functions for splitting data and scaling/normalization.

Dependencies
------------
- numpy
"""

from typing import Tuple, Optional
import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple train/test split.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    test_size : float
        Fraction of samples assigned to the test set.
    shuffle : bool
        Whether to shuffle before splitting.
    random_state : int or None
        Seed for shuffling.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)

    n_test = int(np.round(test_size * n))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for standardization (z-score).

    Returns
    -------
    mean : np.ndarray, shape (n_features,)
    std : np.ndarray, shape (n_features,)
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0.0] = 1.0  # avoid divide by zero
    return mean, std


def standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply z-score standardization using provided mean/std.
    """
    X = np.asarray(X, dtype=float)
    return (X - mean) / std


def minmax_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute min and range for min-max scaling to [0, 1].

    Returns
    -------
    min_ : np.ndarray
    range_ : np.ndarray
    """
    X = np.asarray(X, dtype=float)
    min_ = X.min(axis=0)
    max_ = X.max(axis=0)
    range_ = max_ - min_
    range_[range_ == 0.0] = 1.0
    return min_, range_


def minmax_transform(X: np.ndarray, min_: np.ndarray, range_: np.ndarray) -> np.ndarray:
    """
    Apply min-max scaling to [0, 1] using provided min/range.
    """
    X = np.asarray(X, dtype=float)
    return (X - min_) / range_
