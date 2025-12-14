"""
A regression tree is a decision-tree model that predicts continuous values by recursively
splitting the feature space into regions that minimize within-region squared error.
Each leaf node outputs the mean target value of the samples in that region.

Dependencies: numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any

import numpy as np


@dataclass
class _Node:
    """
    Internal tree node used by :class:`regression_tree`.

    Parameters
    ----------
    is_leaf : bool
        Whether this node is a leaf.
    value : float
        Prediction stored at the node (mean of targets at this node).
    feature_index : int or None
        Index of the feature used for splitting.
    threshold : float or None
        Threshold used for splitting.
    left : _Node or None
        Left child node.
    right : _Node or None
        Right child node.
    n_samples : int
        Number of samples reaching this node.
    impurity : float
        Mean squared error (MSE) of targets in this node.
    """

    is_leaf: bool
    value: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    n_samples: int = 0
    impurity: float = 0.0


class regression_tree:
    """
    Regression tree using mean squared error (MSE) as the split criterion.

    The tree is grown greedily in a top-down manner by selecting, at each node,
    the feature and threshold that maximize the reduction in MSE.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. If None, the tree can grow until other stopping
        criteria are met.
    min_samples_split : int, default=2
        Minimum number of samples required to attempt a split at a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required in a leaf.
    min_impurity_decrease : float, default=0.0
        Minimum MSE reduction required to perform a split.
    max_features : int, float, {"sqrt", "log2"} or None, default=None
        Number of features to consider at each split.
    random_state : int or None, default=None
        Random seed for feature subsampling.

    Attributes
    ----------
    root_ : _Node or None
        Root node of the trained tree. Set after calling :meth:`fit`.
    n_features_in_ : int or None
        Number of features seen during :meth:`fit`.
    feature_importances_ : numpy.ndarray or None
        Normalized impurity-decrease-based importances, shape (n_features_in_,).
        Set after calling :meth:`fit`.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the regression tree with parameter validation.

        Parameters
        ----------
        max_depth : int or None, default=None
            Maximum depth of the tree.
        min_samples_split : int, default=2
            Minimum number of samples required to attempt a split.
        min_samples_leaf : int, default=1
            Minimum number of samples required in a leaf.
        min_impurity_decrease : float, default=0.0
            Minimum MSE reduction required to perform a split.
        max_features : int, float, {"sqrt", "log2"} or None, default=None
            Number of features to consider at each split.
        random_state : int or None, default=None
            Random seed for feature subsampling.

        Raises
        ------
        TypeError
            If any parameter is of an invalid type.
        ValueError
            If any parameter is outside valid bounds.
        """
        self._validate_init_params(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            random_state=random_state,
        )

        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Optional[_Node] = None
        self.n_features_in_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None

        self._rng = np.random.default_rng(random_state)
        self._feature_importance_raw: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "regression_tree":
        """
        Fit the regression tree to training data.

        Parameters
        ----------
        X : array-like
            Feature matrix of shape (n_samples, n_features). Accepts numpy arrays
            or array-like objects convertible to numpy arrays.
        y : array-like
            Target vector of shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        regression_tree
            Fitted model.

        Raises
        ------
        TypeError
            If X or y are not array-like / convertible to numpy arrays.
        ValueError
            If shapes are invalid or inputs contain NaN/inf.
        """
        X, y = self._validate_X_y(X, y)
        self.n_features_in_ = int(X.shape[1])
        self._feature_importance_raw = np.zeros(self.n_features_in_, dtype=float)

        self.root_ = self._build_tree(X, y, depth=0)

        total = float(self._feature_importance_raw.sum())
        if total > 0.0:
            self.feature_importances_ = self._feature_importance_raw / total
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict target values for samples.

        Parameters
        ----------
        X : array-like
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted values of shape (n_samples,).

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        TypeError
            If X is not array-like / convertible to numpy array.
        ValueError
            If X has invalid shape or contains NaN/inf.
        """
        if self.root_ is None or self.n_features_in_ is None:
            raise RuntimeError("Model must be fit before prediction.")

        X = self._validate_X(X)
        preds = np.empty(X.shape[0], dtype=float)
        for i, xi in enumerate(X):
            preds[i] = self._predict_one(self.root_, xi)
        return preds

    def score(self, X: Any, y: Any) -> float:
        """
        Compute the coefficient of determination R^2.

        Parameters
        ----------
        X : array-like
            Feature matrix of shape (n_samples, n_features).
        y : array-like
            True target values of shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        float
            R^2 score.

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        TypeError
            If inputs are not array-like / convertible to numpy arrays.
        ValueError
            If shapes are invalid or inputs contain NaN/inf.
        """
        if self.root_ is None:
            raise RuntimeError("Model must be fit before scoring.")

        X, y = self._validate_X_y(X, y)
        y_pred = self.predict(X)

        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    # ------------------------- Validation helpers -------------------------

    def _validate_init_params(
        self,
        *,
        max_depth: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        max_features: Optional[Union[int, float, str]],
        random_state: Optional[int],
    ) -> None:
        """
        Validate constructor parameters with type and bounds checks.

        Parameters
        ----------
        max_depth : int or None
            Maximum depth.
        min_samples_split : int
            Minimum samples to split.
        min_samples_leaf : int
            Minimum samples per leaf.
        min_impurity_decrease : float
            Minimum impurity decrease.
        max_features : int, float, {"sqrt","log2"} or None
            Feature subsampling rule.
        random_state : int or None
            Random seed.

        Raises
        ------
        TypeError
            If parameter types are invalid.
        ValueError
            If parameter values are invalid.
        """
        # max_depth
        if max_depth is not None:
            if not isinstance(max_depth, (int, np.integer)) or isinstance(max_depth, bool):
                raise TypeError("max_depth must be an int or None.")
            if int(max_depth) < 0:
                raise ValueError("max_depth must be >= 0 or None.")

        # min_samples_split
        if not isinstance(min_samples_split, (int, np.integer)) or isinstance(min_samples_split, bool):
            raise TypeError("min_samples_split must be an int.")
        if int(min_samples_split) < 2:
            raise ValueError("min_samples_split must be >= 2.")

        # min_samples_leaf
        if not isinstance(min_samples_leaf, (int, np.integer)) or isinstance(min_samples_leaf, bool):
            raise TypeError("min_samples_leaf must be an int.")
        if int(min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf must be >= 1.")


        # min_impurity_decrease
        if not isinstance(min_impurity_decrease, (int, float, np.integer, np.floating)) or isinstance(
            min_impurity_decrease, bool
        ):
            raise TypeError("min_impurity_decrease must be a float.")
        if float(min_impurity_decrease) < 0.0:
            raise ValueError("min_impurity_decrease must be >= 0.0.")

        # max_features
        if max_features is not None:
            if isinstance(max_features, str):
                if max_features not in {"sqrt", "log2"}:
                    raise ValueError("max_features string must be one of {'sqrt', 'log2'}.")
            elif isinstance(max_features, (int, np.integer)) and not isinstance(max_features, bool):
                if int(max_features) < 1:
                    raise ValueError("max_features int must be >= 1.")
            elif isinstance(max_features, (float, np.floating)):
                mf = float(max_features)
                if not (0.0 < mf <= 1.0):
                    raise ValueError("max_features float must be in (0, 1].")
            else:
                raise TypeError("max_features must be None, int, float, 'sqrt', or 'log2'.")

        # random_state
        if random_state is not None:
            if not isinstance(random_state, (int, np.integer)) or isinstance(random_state, bool):
                raise TypeError("random_state must be an int or None.")

    def _validate_X_y(self, X: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and normalize feature matrix and target vector.

        Parameters
        ----------
        X : array-like
            Candidate feature matrix.
        y : array-like
            Candidate target vector.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Validated X (2D float) and y (1D float).

        Raises
        ------
        TypeError
            If X or y are not array-like / convertible to numpy arrays.
        ValueError
            If shapes are incompatible or contain NaN/inf.
        """
        X = self._to_float_array(X, name="X")
        y = self._to_float_array(y, name="y")

        if X.ndim != 2:
            raise ValueError("X must be 2D with shape (n_samples, n_features).")

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError("y must be 1D or a 2D column vector of shape (n_samples, 1).")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows (samples).")

        if X.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinite values.")

        return X, y.astype(float, copy=False)

    def _validate_X(self, X: Any) -> np.ndarray:
        """
        Validate feature matrix for prediction.

        Parameters
        ----------
        X : array-like
            Candidate feature matrix.

        Returns
        -------
        numpy.ndarray
            Validated 2D float feature matrix.

        Raises
        ------
        TypeError
            If X is not array-like / convertible to numpy array.
        ValueError
            If X has invalid dimensionality, feature count mismatch, or contains NaN/inf.
        """
        if self.n_features_in_ is None:
            raise RuntimeError("Model must be fit before validating X.")

        X = self._to_float_array(X, name="X")
        if X.ndim != 2:
            raise ValueError("X must be 2D with shape (n_samples, n_features).")
        if X.shape[1] != int(self.n_features_in_):
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fit with {self.n_features_in_} features."
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        return X

    def _to_float_array(self, a: Any, *, name: str) -> np.ndarray:
        """
        Convert input to a numpy float array with helpful errors.

        Parameters
        ----------
        a : any
            Input object to convert.
        name : str
            Name used in error messages.

        Returns
        -------
        numpy.ndarray
            Numpy array with dtype float (copy avoided when possible).

        Raises
        ------
        TypeError
            If the input cannot be converted to a numeric array.
        """
        try:
            arr = np.asarray(a)
        except Exception as e:
            raise TypeError(f"{name} must be array-like and convertible to a numpy array.") from e

        # Reject object arrays (often strings/mixed types) early.
        if arr.dtype == object:
            # Allow if it can safely be cast to float (e.g., list of numeric strings),
            # otherwise raise a clear TypeError.
            try:
                arr = arr.astype(float)
            except Exception as e:
                raise TypeError(f"{name} must contain numeric values.") from e
        else:
            # Regular numeric dtypes: cast to float without unnecessary copies.
            try:
                arr = arr.astype(float, copy=False)
            except Exception as e:
                raise TypeError(f"{name} must contain numeric values.") from e

        return arr

    # ------------------------- Core tree logic -------------------------

    def _mse(self, y: np.ndarray) -> float:
        """
        Compute mean squared error (variance around the mean) of a target vector.

        Parameters
        ----------
        y : numpy.ndarray
            Target values.

        Returns
        -------
        float
            Mean squared error.
        """
        if y.size == 0:
            return 0.0
        mu = float(np.mean(y))
        return float(np.mean((y - mu) ** 2))

    def _sample_feature_indices(self, n_features: int) -> np.ndarray:
        """
        Select feature indices to consider for splitting at a node.

        Parameters
        ----------
        n_features : int
            Total number of features available.

        Returns
        -------
        numpy.ndarray
            Array of selected feature indices.
        """
        mf = self.max_features
        if mf is None:
            return np.arange(n_features, dtype=int)

        if isinstance(mf, str):
            if mf == "sqrt":
                k = int(np.ceil(np.sqrt(n_features)))
            else:  # "log2"
                k = int(np.ceil(np.log2(max(n_features, 2))))
        elif isinstance(mf, (int, np.integer)):
            k = int(mf)
        else:  # float
            k = int(np.ceil(float(mf) * n_features))

        k = max(1, min(n_features, k))
        return self._rng.choice(n_features, size=k, replace=False)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Determine the best feature and threshold to split on.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix for samples in a node, shape (n_samples, n_features).
        y : numpy.ndarray
            Target values for samples in a node, shape (n_samples,).

        Returns
        -------
        feature_index : int or None
            Index of best feature, or None if no valid split is found.
        threshold : float or None
            Threshold for best split, or None if no valid split is found.
        impurity_decrease : float
            Reduction in MSE achieved by the split (0.0 if no valid split is found).
        """
        n_samples, n_features = X.shape
        parent_impurity = self._mse(y)

        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        best_gain = 0.0

        features = self._sample_feature_indices(n_features)

        for j in features:
            order = np.argsort(X[:, j], kind="mergesort")
            x_sorted = X[order, j]
            y_sorted = y[order]

            # Candidate splits between distinct adjacent feature values.
            split_positions = np.where(x_sorted[1:] != x_sorted[:-1])[0]
            if split_positions.size == 0:
                continue

            # Prefix sums for SSE computation.
            y_cumsum = np.cumsum(y_sorted)
            y2_cumsum = np.cumsum(y_sorted ** 2)
            total_sum = y_cumsum[-1]
            total_sum2 = y2_cumsum[-1]

            for idx in split_positions:
                n_left = idx + 1
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_sum = y_cumsum[idx]
                left_sum2 = y2_cumsum[idx]
                right_sum = total_sum - left_sum
                right_sum2 = total_sum2 - left_sum2

                left_sse = left_sum2 - (left_sum * left_sum) / n_left
                right_sse = right_sum2 - (right_sum * right_sum) / n_right

                weighted_mse = (left_sse + right_sse) / n_samples
                gain = parent_impurity - weighted_mse

                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = int(j)
                    best_threshold = float((x_sorted[idx] + x_sorted[idx + 1]) / 2.0)

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """
        Recursively construct the regression tree.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix for the current node.
        y : numpy.ndarray
            Target values for the current node.
        depth : int
            Current tree depth (root is 0).

        Returns
        -------
        _Node
            Root node of the constructed subtree.
        """
        n = int(y.size)
        value = float(np.mean(y))
        impurity = self._mse(y)

        # Stopping criteria
        if n < self.min_samples_split:
            return _Node(True, value, n_samples=n, impurity=impurity)
        if self.max_depth is not None and depth >= int(self.max_depth):
            return _Node(True, value, n_samples=n, impurity=impurity)
        if float(np.max(y) - np.min(y)) == 0.0:
            return _Node(True, value, n_samples=n, impurity=impurity)

        feature, threshold, gain = self._best_split(X, y)

        if feature is None or threshold is None:
            return _Node(True, value, n_samples=n, impurity=impurity)
        if gain < self.min_impurity_decrease:
            return _Node(True, value, n_samples=n, impurity=impurity)

        mask = X[:, feature] <= threshold
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        # Final safety check
        if y_left.size < self.min_samples_leaf or y_right.size < self.min_samples_leaf:
            return _Node(True, value, n_samples=n, impurity=impurity)

        # Accumulate impurity decrease for importances
        if self._feature_importance_raw is not None:
            self._feature_importance_raw[feature] += gain * n

        left = self._build_tree(X_left, y_left, depth + 1)
        right = self._build_tree(X_right, y_right, depth + 1)

        return _Node(
            False,
            value,
            feature_index=feature,
            threshold=threshold,
            left=left,
            right=right,
            n_samples=n,
            impurity=impurity,
        )

    def _predict_one(self, node: _Node, x: np.ndarray) -> float:
        """
        Predict for a single sample by traversing the tree.

        Parameters
        ----------
        node : _Node
            Current node.
        x : numpy.ndarray
            Feature vector of shape (n_features,).

        Returns
        -------
        float
            Predicted value.
        """
        while not node.is_leaf:
            # Defensive checks in case of inconsistent node state.
            if node.feature_index is None or node.threshold is None or node.left is None or node.right is None:
                return float(node.value)

            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return float(node.value)


if __name__ == "__main__":
    """
    Minimal usage example.

    Notes
    -----
    This block is only for quick manual testing. Remove or modify as needed.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * rng.normal(size=200)

    tree = regression_tree(max_depth=4, min_samples_leaf=5, random_state=0, max_features="sqrt")
    tree.fit(X, y)

    print("R^2:", tree.score(X, y))
    print("Feature importances:", tree.feature_importances_)
