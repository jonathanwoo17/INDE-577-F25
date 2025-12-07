from typing import Callable, Optional
import numpy as np
from rice_ml.supervised_learning.metrics import euclidean_distance, manhattan_distance

"""
k_means_clustering.py
---------
This module implements the k-means clustering algorithm using NumPy.

Features
--------
- Arbitrary-dimensional data.
- Supports Euclidean or Manhattan distance.
- Customizable iteration limit and convergence tolerance.
- Class-based design consistent with DBScan style.

Dependencies
------------
- numpy
- rice_ml.supervised_learning.metrics
"""


class k_means_clustering:
    """
    Simple k-means clustering implementation with 'euclidean' and 'manhattan'
    distance metrics.

    Parameters
    ----------
    n_clusters : int
        The number of clusters (k). Must be >= 1. (Required)
    metric : {'euclidean', 'manhattan'}, optional
        Distance metric to use. Default is 'euclidean'.
    max_iter : int, optional
        Maximum number of iterations allowed. Default is 300.
    tol : float, optional
        Tolerance for convergence based on centroid movement. Default is 1e-4.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Cluster index assigned to each point.
    n_iter_ : int
        Number of iterations run before stopping.
    """

    def __init__(
        self,
        n_clusters: int,
        metric: str = "euclidean",
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> None:
        """
        Initialize the k_means_clustering instance.

        Parameters
        ----------
        n_clusters : int
            The number of clusters (k). Must be >= 1.
        metric : {'euclidean', 'manhattan'}, optional
            Distance metric to use. Default is 'euclidean'.
        max_iter : int, optional
            Maximum number of iterations allowed. Default is 300.
        tol : float, optional
            Tolerance for convergence based on centroid movement. Default is 1e-4.

        Raises
        ------
        TypeError
            If any of the arguments has an incorrect type.
        ValueError
            If n_clusters is not a positive integer.
        """
        # Hard-coded type checks
        if not isinstance(n_clusters, int):
            raise TypeError(
                f"Expected int for n_clusters, got {type(n_clusters).__name__}"
            )
        if not isinstance(metric, str):
            raise TypeError(
                f"Expected str for metric, got {type(metric).__name__}"
            )
        if not isinstance(max_iter, int):
            raise TypeError(
                f"Expected int for max_iter, got {type(max_iter).__name__}"
            )
        if not isinstance(tol, float):
            raise TypeError(
                f"Expected float for tol, got {type(tol).__name__}"
            )

        if n_clusters is None:
            raise ValueError("n_clusters must be provided and cannot be optional.")
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")

        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.n_iter_: Optional[int] = None

        self._set_distance_function(metric)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "k_means_clustering":
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        self : k_means_clustering
            Fitted estimator.

        Raises
        ------
        ValueError
            If X is not 2D or if n_clusters exceeds the number of samples.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features).")

        n_samples = X.shape[0]
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed number of samples.")

        # Random initialization of centroids (no random_state)
        initial_indices = np.random.choice(
            n_samples,
            size=self.n_clusters,
            replace=False,
        )
        centroids = X[initial_indices].copy()

        for iteration in range(1, self.max_iter + 1):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._compute_centroids(X, labels, X.shape[1])
            shift = self._max_centroid_shift(centroids, new_centroids)

            if shift < self.tol:
                centroids = new_centroids
                self.n_iter_ = iteration
                break

            centroids = new_centroids

        if self.n_iter_ is None:
            self.n_iter_ = self.max_iter

        self.cluster_centers_ = centroids
        self.labels_ = self._assign_clusters(X, centroids)
        
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit k-means clustering and return cluster labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster index of each sample.
        """
        self.fit(X)
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign clusters to new samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("k_means_clustering has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        return self._assign_clusters(X, self.cluster_centers_)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _set_distance_function(self, metric: str) -> None:
        """
        Set the distance function based on the chosen metric.

        Parameters
        ----------
        metric : {'euclidean', 'manhattan'}

        Raises
        ------
        ValueError
            If metric is not 'euclidean' or 'manhattan'.
        """
        m = metric.lower()
        if m == "euclidean":
            self._distance = euclidean_distance
        elif m == "manhattan":
            self._distance = manhattan_distance
        else:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each sample in X to the nearest centroid.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data points to assign.
        centroids : ndarray of shape (n_clusters, n_features)
            Current cluster centers.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the nearest centroid for each sample.
        """
        labels = np.empty(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            dists = np.array([self._distance(X[i], c) for c in centroids])
            labels[i] = int(np.argmin(dists))
        return labels

    def _compute_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        """
        Compute new centroids as the mean of assigned samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data points.
        labels : ndarray of shape (n_samples,)
            Cluster assignment for each sample.
        n_features : int
            Number of features.

        Returns
        -------
        new_centroids : ndarray of shape (n_clusters, n_features)
            Updated cluster centers.
        """
        new_centroids = np.zeros((self.n_clusters, n_features))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if cluster_points.size > 0:
                new_centroids[k] = cluster_points.mean(axis=0)

        return new_centroids

    def _max_centroid_shift(
        self,
        old: np.ndarray,
        new: np.ndarray,
    ) -> float:
        """
        Return the maximum centroid movement between iterations.

        Parameters
        ----------
        old : ndarray of shape (n_clusters, n_features)
            Previous cluster centers.
        new : ndarray of shape (n_clusters, n_features)
            Updated cluster centers.

        Returns
        -------
        shift : float
            Maximum distance between corresponding old and new centroids.
        """
        shifts = [self._distance(o, n) for o, n in zip(old, new)]
        return float(max(shifts)) if shifts else 0.0


