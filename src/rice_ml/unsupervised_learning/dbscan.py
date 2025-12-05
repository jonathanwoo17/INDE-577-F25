from typing import Callable, Optional
import numpy as np
from rice_ml.supervised_learning.metrics import euclidean_distance, manhattan_distance
"""
dbscan.py
---------
This module impelments the DBScan algorithm from scratch using NumPy. DBScan clusters data into a non-user
specified number of clusters and can detect non-spherical clusters. 

Dependencies
------------
- numpy
- pandas
- matplotlib
"""



class DBScan:
    """
    Simple DBSCAN implementation supporting 'euclidean' and 'manhattan'
    distance metrics.

    This version:
    - Uses 1-indexed cluster labels: 1, 2, ..., n_clusters.
    - Assigns all outliers (noise points) the label n_clusters + 1.
    - Returns both the labels and the number of clusters from `fit_predict`.

    Parameters
    ----------
    eps : float, optional
        Neighborhood radius. Default is 0.5.
    min_samples : int, optional
        Minimum number of samples required to form a dense region (core point).
        Default is 5.
    metric : {'euclidean', 'manhattan'}, optional
        Distance metric to use. Default is 'euclidean'.

    Attributes
    ----------
    eps_ : float
        Effective neighborhood radius used during fitting.
    min_samples_ : int
        Effective minimum number of samples used during fitting.
    labels_ : ndarray of shape (n_samples,)
        Final labels for each point. Cluster labels are 1..n_clusters,
        and outliers have label n_clusters + 1.
    n_clusters_ : int
        Number of clusters found (excluding outliers).
    noise_label_ : int
        The label used for outliers (always 0).
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean") -> None:
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
        Perform DBSCAN clustering on dataset X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
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
        Fit DBSCAN on X and return (labels, n_clusters).

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            1..n_clusters for real clusters, 0 for outliers
        n_clusters : int
            Number of real clusters found (excluding outliers)
        """
        self.fit(X)
        return self.labels_, self.n_clusters_

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _region_query(self, X: np.ndarray, idx: int) -> np.ndarray:
        """
        Return indices of all points within eps of X[idx].
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
        Grow the cluster by exploring density-reachable neighbors.
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