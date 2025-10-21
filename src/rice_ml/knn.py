"""
knn.py
------

K-Nearest Neighbors (KNN) implementation supporting both classification and
regression. Distance metric and weighting scheme are configurable.

Dependencies
------------
- numpy
- functions from metrics.py and postprocess.py
"""

from typing import Callable, Literal, Optional, Tuple
import numpy as np

from .metrics import euclidean_distance, manhattan_distance, pairwise_distances
from .postprocess import majority_vote, average_label


DistanceName = Literal["euclidean", "manhattan"]
TaskName = Literal["classification", "regression"]
WeightName = Literal["uniform", "distance"]


_DISTANCE_REGISTRY: dict[DistanceName, Callable[[np.ndarray, np.ndarray], float]] = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
}


class KNN:
    """
    K-Nearest Neighbors estimator.

    Parameters
    ----------
    k : int
        Number of neighbors to use.
    task : {"classification", "regression"}
        Type of prediction task.
    distance : {"euclidean", "manhattan"} or callable
        Distance metric to use. If callable, it must accept (x, y) -> float.
    weights : {"uniform", "distance"}
        Weighting scheme. "uniform" = equal weights; "distance" = 1/(d + eps).
    eps : float
        Small constant to stabilize distance-based weights.
    return_neighbors : bool
        If True, predict(...) also returns (indices, distances).

    Attributes
    ----------
    X_ : np.ndarray, shape (n_samples, n_features)
        Training features.
    y_ : np.ndarray, shape (n_samples,)
        Training targets/labels.
    """

    def __init__(
        self,
        k: int = 5,
        task: TaskName = "classification",
        distance: DistanceName | Callable[[np.ndarray, np.ndarray], float] = "euclidean",
        weights: WeightName = "uniform",
        eps: float = 1e-12,
        return_neighbors: bool = False,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'.")
        if isinstance(distance, str) and distance not in _DISTANCE_REGISTRY:
            raise ValueError(f"Unknown distance '{distance}'.")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'.")

        self.k = int(k)
        self.task = task
        self.distance = (
            _DISTANCE_REGISTRY[distance] if isinstance(distance, str) else distance
        )
        self.weights = weights
        self.eps = float(eps)
        self.return_neighbors = return_neighbors

        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """
        Store the training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self : KNN
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D and match the number of rows in X.")
        if self.k > X.shape[0]:
            raise ValueError("k cannot exceed the number of training samples.")

        self.X_ = X
        self.y_ = y
        return self

    def predict(
        self, X: np.ndarray
    ) -> np.ndarray | Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict labels/targets for input X.

        Returns
        -------
        preds : np.ndarray, shape (n_samples,)
            Predicted labels (classification) or values (regression).
        (idxs, dists) : tuple of arrays (optional)
            If return_neighbors=True, also returns indices and distances for
            the selected neighbors per row.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        idxs, dists = self.kneighbors(X, n_neighbors=self.k, return_distance=True)

        if self.weights == "uniform":
            weights = np.ones_like(dists, dtype=float)
        else:
            # distance weighting; handle zeros (exact matches)
            weights = 1.0 / (dists + self.eps)

        # Gather neighbor labels/targets
        y_neighbors = self.y_[idxs]

        if self.task == "classification":
            preds = np.array(
                [majority_vote(y_row, w_row) for y_row, w_row in zip(y_neighbors, weights)]
            )
        else:  # regression
            preds = np.array(
                [average_label(y_row.astype(float), w_row) for y_row, w_row in zip(y_neighbors, weights)]
            )

        if self.return_neighbors:
            return preds, (idxs, dists)
        return preds

    def kneighbors(
        self,
        X: np.ndarray,
        n_neighbors: Optional[int] = None,
        return_distance: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Compute k nearest neighbors for each row in X.

        Parameters
        ----------
        X : np.ndarray, shape (n_query, n_features)
        n_neighbors : int or None
            Number of neighbors to retrieve; defaults to self.k.
        return_distance : bool
            If True, also return the distances.

        Returns
        -------
        idxs : np.ndarray, shape (n_query, k)
            Indices of neighbors in the training set.
        dists : np.ndarray, shape (n_query, k)  (optional)
            Distances to those neighbors.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        k = self.k if n_neighbors is None else int(n_neighbors)
        if k <= 0:
            raise ValueError("n_neighbors must be positive.")
        if k > self.X_.shape[0]:
            raise ValueError("n_neighbors cannot exceed number of training samples.")

        # Brute-force distance computation
        dmat = pairwise_distances(X, self.X_, metric=self.distance)
        idxs = np.argpartition(dmat, kth=k - 1, axis=1)[:, :k]

        # Ensure sorted by distance within the k chunk
        row_indices = np.arange(X.shape[0])[:, None]
        d_k = np.take_along_axis(dmat, idxs, axis=1)
        order = np.argsort(d_k, axis=1)
        idxs = np.take_along_axis(idxs, order, axis=1)
        d_k = np.take_along_axis(d_k, order, axis=1)

        if return_distance:
            return idxs, d_k
        return idxs

    # ------------------------ Internal ------------------------

    def _check_is_fitted(self) -> None:
        if self.X_ is None or self.y_ is None:
            raise RuntimeError("Estimator is not fitted. Call fit(X, y) first.")

