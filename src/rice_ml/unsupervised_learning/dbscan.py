from typing import Callable, Optional
import numpy as np
from rice_ml.supervised_learning.metrics import euclidean_distance, manhattan_distance
"""
dbscan.py
---------

A from-scratch implementation of the DBSCAN (Density-Based Spatial Clustering of
Applications with Noise) algorithm using NumPy.

DBSCAN groups points into clusters based on local density. It can discover
non-spherical clusters and does not require specifying the number of clusters
in advance. Points that do not belong to any cluster are labeled as noise.

Notes
-----
This implementation:
- Produces 1-indexed cluster labels: 1, 2, ..., n_clusters.
- Assigns all outliers (noise points) the label 0.
- Supports two distance metrics: 'euclidean' and 'manhattan'.

Dependencies
------------
- numpy
- typing
- euclidean_distance, manhattan_distance from rice_ml.supervised_learning.metrics
"""



class DBScan:
    """
    Density-based clustering via DBSCAN.

    This class implements DBSCAN clustering with support for Euclidean and
    Manhattan distance metrics. Clusters are assigned integer labels starting
    at 1, while noise points retain label 0.

    Parameters
    ----------
    eps : float, optional
        Neighborhood radius. Two points are considered neighbors if their
        distance is less than or equal to `eps`. Default is 0.5.
    min_samples : int, optional
        Minimum number of points in an `eps`-neighborhood for a point to be
        considered a core point. Default is 5.
    metric : {'euclidean', 'manhattan'}, optional
        Distance metric to use. Default is 'euclidean'.

    Attributes
    ----------
    eps : float
        User-provided neighborhood radius.
    min_samples : int
        User-provided minimum samples threshold.
    metric : {'euclidean', 'manhattan'}
        User-provided distance metric name.
    eps_ : float or None
        Effective neighborhood radius used during fitting (set in `fit`).
    min_samples_ : int or None
        Effective minimum number of samples used during fitting (set in `fit`).
    labels_ : ndarray of shape (n_samples,) or None
        Final labels for each sample after fitting.

        - Cluster labels are in ``{1, 2, ..., n_clusters_}``.
        - Noise (outliers) are labeled ``0``.
    n_clusters_ : int or None
        Number of clusters found (excluding noise).
    noise_label_ : int or None
        The label used for outliers (always 0 after fitting).
    _distance : Callable[[ndarray, ndarray], float]
        Distance function chosen from the provided metric (internal).

    See Also
    --------
    rice_ml.supervised_learning.metrics.euclidean_distance
    rice_ml.supervised_learning.metrics.manhattan_distance
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean") -> None:
        """
        Initialize a DBScan instance.

        Parameters
        ----------
        eps : float, optional
            Neighborhood radius used to determine whether points are neighbors.
            Default is 0.5.
        min_samples : int, optional
            Minimum number of points in the neighborhood required for a point
            to be a core point. Default is 5.
        metric : {'euclidean', 'manhattan'}, optional
            Distance metric used to compute pairwise distances.
            Default is 'euclidean'.

        Raises
        ------
        ValueError
            If `metric` is not one of {'euclidean', 'manhattan'}.
        TypeError
            If `min_samples` is not an integer.

        Notes
        -----
        The fitted attributes `eps_`, `min_samples_`, `labels_`, `n_clusters_`,
        and `noise_label_` are initialized to None and populated during `fit`.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        # Filled in during fit
        self.eps_: Optional[float] = None
        self.min_samples_: Optional[int] = None
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None
        self.noise_label_: Optional[int] = None

        # Choose distance function based on metric
        if metric == "euclidean":
            self._distance: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance
        elif metric == "manhattan":
            self._distance: Callable[[np.ndarray, np.ndarray], float] = manhattan_distance
        else:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")

        # Simple type check for min_samples
        if not isinstance(min_samples, int):
            raise TypeError(f"Expected int for min_samples, got {type(min_samples).__name__}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "DBScan":
        """
        Compute DBSCAN clustering from features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix where each row corresponds to a sample and each
            column corresponds to a feature.

        Returns
        -------
        self : DBScan
            Fitted estimator with `labels_`, `n_clusters_`, and `noise_label_`
            populated.

        Raises
        ------
        ValueError
            If `X` is not a 2D array with shape (n_samples, n_features).

        Notes
        -----
        This method follows the standard DBSCAN procedure:

        1. Iterate through points; if a point is unvisited, mark it visited and
           find its `eps`-neighbors.
        2. If the point has fewer than `min_samples` neighbors, it remains noise
           (label 0).
        3. Otherwise, create a new cluster and expand it by exploring all
           density-reachable points.

        Noise points remain labeled 0. Cluster labels start at 1 and increase
        sequentially.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        n_samples = X.shape[0]

        self.eps_ = float(self.eps)
        self.min_samples_ = int(self.min_samples)

        # Internal labels: 0 means unassigned/noise initially
        labels = np.zeros(n_samples, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0  # will become 1..n_clusters

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._region_query(X, i)

            if neighbors.size < self.min_samples_:
                # Not enough neighbors -> noise (label stays 0)
                continue

            # Found a new cluster
            cluster_id += 1
            labels[i] = cluster_id

            self._expand_cluster(
                X=X,
                labels=labels,
                visited=visited,
                neighbors=neighbors,
                cluster_id=cluster_id,
            )

        # Outliers remain labeled 0
        self.labels_ = labels
        self.n_clusters_ = cluster_id  # real clusters only
        self.noise_label_ = 0

        return self

    def fit_predict(self, X: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Fit DBSCAN on `X` and return the resulting labels and number of clusters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each sample.

            - Values in ``{1, 2, ..., n_clusters}`` indicate cluster membership.
            - Value ``0`` indicates noise (outlier).
        n_clusters : int
            Number of clusters found (excluding noise).

        See Also
        --------
        fit : Computes and stores the fitted clustering results.
        """
        self.fit(X)
        return self.labels_, self.n_clusters_

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _region_query(self, X: np.ndarray, idx: int) -> np.ndarray:
        """
        Find neighbors of a point within the `eps` radius.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix.
        idx : int
            Index of the query point in `X`.

        Returns
        -------
        neighbors : ndarray of shape (n_neighbors,)
            Indices of points whose distance to `X[idx]` is less than or equal
            to `self.eps_`. The returned indices are those produced by
            `np.where(...)` and therefore are sorted in ascending order.

        Notes
        -----
        The query point `idx` is included in the returned neighbor indices
        because the distance from a point to itself is 0, which is always
        <= `eps_` (assuming non-negative `eps_`).

        This method assumes `self.eps_` has been set (i.e., `fit` has started).
        """
        distances = np.array([self._distance(X[idx], X[j]) for j in range(X.shape[0])])
        return np.where(distances <= self.eps_)[0]

    def _expand_cluster(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        visited: np.ndarray,
        neighbors: np.ndarray,
        cluster_id: int,
    ) -> None:
        """
        Expand a newly created cluster from an initial seed set of neighbors.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix.
        labels : ndarray of shape (n_samples,)
            Current label assignment array. Modified in-place. Points labeled 0
            are considered unassigned/noise and may be assigned to `cluster_id`.
        visited : ndarray of shape (n_samples,)
            Boolean array tracking which points have been visited. Modified
            in-place.
        neighbors : ndarray of shape (n_neighbors,)
            Seed neighbor indices for the initial core point that started the
            cluster.
        cluster_id : int
            The label of the cluster being expanded (1-indexed).

        Returns
        -------
        None

        Notes
        -----
        This method performs the DBSCAN "density-reachability" expansion:

        - It maintains a queue initialized with `neighbors`.
        - For each point popped from the queue:
          - If unvisited, mark visited and compute its neighbors.
          - If it is a core point (enough neighbors), append its neighbors to
            the queue (avoiding duplicates already in the queue).
          - If it is currently unlabeled/noise (label 0), assign it to the
            current cluster.

        Labels are updated in-place; the method does not return anything.
        """
        queue = list(neighbors)

        while queue:
            idx = queue.pop()

            if not visited[idx]:
                visited[idx] = True
                new_neighbors = self._region_query(X, idx)

                # If idx is a core point, add neighbors to the queue
                if new_neighbors.size >= self.min_samples_:
                    for n in new_neighbors:
                        if n not in queue:
                            queue.append(n)

            # Assign this point to cluster if it isn't assigned yet
            if labels[idx] == 0:
                labels[idx] = cluster_id
