"""
postprocess.py
--------------

Neighbor aggregation functions:
- majority_vote: for classification
- average_label: for regression

Dependencies
------------
- numpy
"""

from typing import Optional
import numpy as np


def majority_vote(
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Weighted majority vote with deterministic tie-break.

    If weights is None, uses uniform weights. In case of a tie, the class with
    the smallest value (after np.unique sorting) is chosen to keep it deterministic.

    Parameters
    ----------
    labels : np.ndarray, shape (k,)
        Neighbor labels (can be numeric or strings).
    weights : np.ndarray or None, shape (k,)
        Optional weights for each neighbor (e.g., distance-based).

    Returns
    -------
    winner : same dtype as labels
    """
    labels = np.asarray(labels)
    if weights is None:
        weights = np.ones_like(labels, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # Aggregate weights per unique label
    uniq, inv = np.unique(labels, return_inverse=True)
    agg = np.zeros(uniq.shape[0], dtype=float)
    np.add.at(agg, inv, weights)

    # Pick label with maximum aggregated weight; deterministic tie-break by np.argmax order
    idx = int(np.argmax(agg))
    return uniq[idx]


def average_label(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Weighted average for regression targets.

    Parameters
    ----------
    values : np.ndarray, shape (k,)
        Neighbor target values (numeric).
    weights : np.ndarray or None, shape (k,)
        Optional weights for each neighbor.

    Returns
    -------
    avg : float
    """
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(values.mean())
    weights = np.asarray(weights, dtype=float)
    wsum = float(weights.sum())
    if wsum == 0.0:
        return float(values.mean())
    return float(np.dot(weights, values) / wsum)
