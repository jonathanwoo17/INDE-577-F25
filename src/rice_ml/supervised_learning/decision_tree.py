"""
Decision Tree Classifier (Entropy / Information Gain)

This module implements a decision tree classifier from scratch for supervised
learning on purely numeric datasets. The tree uses entropy as its impurity
measure and selects splits by maximizing information gain. Continuous features
are handled via binary threshold splits.

Key characteristics
-------------------
- Supports only numeric feature matrices (NumPy arrays)
- Uses entropy (Shannon entropy) and information gain for split selection
- Performs binary splits of the form x_j <= threshold vs x_j > threshold

The implementation includes:
- Recursive tree construction with configurable stopping criteria
- Feature subsampling support (useful for random-forest-style experiments)
- Prediction of class labels and class probabilities
- Information-gain-based feature importance estimation
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray]


@dataclass
class _Node:
    """Internal tree node.

    Parameters
    ----------
    is_leaf
        Whether this node is a leaf.
    prediction
        Predicted class label at this node (majority class of training samples reaching the node).
    feature_index
        Feature index used for the split. ``None`` for leaves.
    threshold
        Threshold value for the split. ``None`` for leaves.
    left
        Left child node for samples with ``X[:, feature_index] <= threshold``.
    right
        Right child node for samples with ``X[:, feature_index] > threshold``.
    n_samples
        Number of training samples that reached this node.
    class_counts
        Mapping from class label to count at this node.
    impurity
        Entropy at this node.
    info_gain
        Information gain achieved by the split at this node (0 for leaves).
    """

    is_leaf: bool
    prediction: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    n_samples: int = 0
    class_counts: Optional[Dict[float, int]] = None
    impurity: float = 0.0
    info_gain: float = 0.0


class decision_tree:
    """Decision tree classifier for numeric data using entropy / information gain.

    This is a minimal, readable implementation of a CART-style binary tree that chooses
    splits using entropy-based information gain (ID3-style criterion), while supporting
    continuous numeric features by searching threshold splits.

    Parameters
    ----------
    max_depth
        Maximum depth of the tree. If ``None``, grow until other stopping criteria apply.
    min_samples_split
        Minimum number of samples required to split an internal node.
    min_samples_leaf
        Minimum number of samples required to be at a leaf node.
    n_features
        Number of features to consider when looking for the best split at each node.
        If ``None``, all features are considered.
    random_state
        Random seed for feature subsampling when ``n_features`` is not ``None``.
    min_impurity_decrease
        A node will be split if the best split yields at least this information gain.

    Attributes
    ----------
    n_features_in_
        Number of input features seen during ``fit``.
    classes_
        Sorted unique class labels observed during ``fit``.
    root_
        Root node of the learned decision tree.
    feature_importances_
        Importance scores per feature based on total information gain (normalized to sum to 1).
        Only available after ``fit``.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_features: Optional[int] = None,
        random_state: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        """Initialize the decision tree classifier.

        Parameters
        ----------
        max_depth
            Maximum depth of the tree. If ``None``, the depth is unbounded.
        min_samples_split
            Minimum number of samples required to attempt a split.
        min_samples_leaf
            Minimum number of samples that must remain in each leaf.
        n_features
            Number of candidate features to evaluate per split. If ``None``, use all.
        random_state
            Random seed for feature subsampling.
        min_impurity_decrease
            Minimum information gain required to split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.random_state = random_state
        self.min_impurity_decrease = float(min_impurity_decrease)

        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.root_: Optional[_Node] = None
        self.feature_importances_: Optional[np.ndarray] = None

        self._rng: np.random.Generator = np.random.default_rng(random_state)

        self._validate_hyperparameters()

    def _validate_hyperparameters(self) -> None:
        """Validate hyperparameters provided at initialization.

        Raises
        ------
        TypeError
            If any hyperparameter has an invalid type.
        ValueError
            If any hyperparameter has an invalid value.
        """
        if self.max_depth is not None and (not isinstance(self.max_depth, int) or self.max_depth < 1):
            raise ValueError("max_depth must be None or a positive integer.")

        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")

        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1.")

        if self.n_features is not None:
            if not isinstance(self.n_features, int) or self.n_features < 1:
                raise ValueError("n_features must be None or an integer >= 1.")

        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError("random_state must be None or an integer.")

        if not isinstance(self.min_impurity_decrease, (int, float)) or self.min_impurity_decrease < 0.0:
            raise ValueError("min_impurity_decrease must be a non-negative float.")

    @staticmethod
    def _check_numeric_matrix(X: ArrayLike, *, name: str = "X") -> np.ndarray:
        """Validate and coerce an input to a 2D numeric numpy array.

        Parameters
        ----------
        X
            Input feature matrix.
        name
            Name of the variable for error messages.

        Returns
        -------
        X_arr
            2D numpy array of dtype float64.

        Raises
        ------
        TypeError
            If X is not a numpy array or cannot be converted to a numeric array.
        ValueError
            If X is not 2D or contains NaN/inf.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray.")

        if X.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got ndim={X.ndim}.")

        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"{name} must have a numeric dtype, got dtype={X.dtype}.")

        X_arr = X.astype(np.float64, copy=False)

        if not np.isfinite(X_arr).all():
            raise ValueError(f"{name} contains NaN or infinite values; please clean/impute first.")

        return X_arr

    @staticmethod
    def _check_numeric_vector(y: ArrayLike, *, name: str = "y") -> np.ndarray:
        """Validate and coerce an input to a 1D numeric numpy array.

        Parameters
        ----------
        y
            Input label vector.
        name
            Name of the variable for error messages.

        Returns
        -------
        y_arr
            1D numpy array of dtype float64.

        Raises
        ------
        TypeError
            If y is not a numpy array or has non-numeric dtype.
        ValueError
            If y is not 1D or contains NaN/inf.
        """
        if not isinstance(y, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray.")

        if y.ndim != 1:
            raise ValueError(f"{name} must be a 1D array, got ndim={y.ndim}.")

        if not np.issubdtype(y.dtype, np.number):
            raise TypeError(f"{name} must have a numeric dtype, got dtype={y.dtype}.")

        y_arr = y.astype(np.float64, copy=False)

        if not np.isfinite(y_arr).all():
            raise ValueError(f"{name} contains NaN or infinite values; please clean/impute first.")

        return y_arr

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        """Compute entropy of a label vector.

        Parameters
        ----------
        y
            Label vector of shape (n_samples,).

        Returns
        -------
        entropy
            Entropy in bits.
        """
        if y.size == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()

        # Avoid log2(0) by selecting only positive probabilities (should already be positive).
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _majority_class(y: np.ndarray) -> float:
        """Return the majority class label.

        Parameters
        ----------
        y
            Label vector of shape (n_samples,).

        Returns
        -------
        label
            Majority label. Ties are broken by choosing the smallest label.
        """
        labels, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        winners = labels[counts == max_count]
        return float(np.min(winners))

    @staticmethod
    def _class_counts(y: np.ndarray) -> Dict[float, int]:
        """Compute class counts for a label vector.

        Parameters
        ----------
        y
            Label vector of shape (n_samples,).

        Returns
        -------
        counts
            Dictionary mapping label to count.
        """
        labels, counts = np.unique(y, return_counts=True)
        return {float(lbl): int(c) for lbl, c in zip(labels, counts)}

    def _information_gain(
        self,
        y_parent: np.ndarray,
        y_left: np.ndarray,
        y_right: np.ndarray,
    ) -> float:
        """Compute information gain from splitting a parent node into two children.

        Parameters
        ----------
        y_parent
            Labels at parent node.
        y_left
            Labels going to left child.
        y_right
            Labels going to right child.

        Returns
        -------
        gain
            Information gain (entropy decrease).
        """
        n = y_parent.size
        if n == 0:
            return 0.0

        h_parent = self._entropy(y_parent)
        n_l = y_left.size
        n_r = y_right.size

        if n_l == 0 or n_r == 0:
            return 0.0

        h_children = (n_l / n) * self._entropy(y_left) + (n_r / n) * self._entropy(y_right)
        return float(h_parent - h_children)

    def _candidate_thresholds(self, x_col: np.ndarray) -> np.ndarray:
        """Generate candidate thresholds for a numeric feature.

        Thresholds are midpoints between consecutive sorted unique feature values.

        Parameters
        ----------
        x_col
            Feature column of shape (n_samples,).

        Returns
        -------
        thresholds
            1D array of candidate thresholds.
        """
        uniq = np.unique(x_col)
        if uniq.size <= 1:
            return np.array([], dtype=np.float64)
        return (uniq[:-1] + uniq[1:]) / 2.0

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best (feature, threshold) split by maximizing information gain.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).
        y
            Label vector of shape (n_samples,).
        feature_indices
            Indices of features to consider.

        Returns
        -------
        best_feature
            Feature index of the best split. ``None`` if no valid split found.
        best_threshold
            Threshold of the best split. ``None`` if no valid split found.
        best_gain
            Information gain of the best split (0 if none found).
        """
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        best_gain: float = 0.0

        for j in feature_indices:
            x_col = X[:, j]
            thresholds = self._candidate_thresholds(x_col)
            if thresholds.size == 0:
                continue

            for thr in thresholds:
                left_mask = x_col <= thr
                right_mask = ~left_mask

                # Enforce leaf size constraints early.
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(j)
                    best_threshold = float(thr)

        return best_feature, best_threshold, best_gain

    def _choose_feature_indices(self) -> np.ndarray:
        """Choose candidate feature indices for splitting at a node.

        Returns
        -------
        feature_indices
            Array of feature indices to consider.
        """
        assert self.n_features_in_ is not None

        if self.n_features is None or self.n_features >= self.n_features_in_:
            return np.arange(self.n_features_in_, dtype=int)

        return self._rng.choice(self.n_features_in_, size=self.n_features, replace=False).astype(int)

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        """Determine whether tree growth should stop at the current node.

        Parameters
        ----------
        y
            Labels at the current node.
        depth
            Current depth (root = 0).

        Returns
        -------
        stop
            ``True`` if growth should stop and the node should be a leaf.
        """
        # Pure node
        if np.unique(y).size == 1:
            return True

        # Depth limit
        if self.max_depth is not None and depth >= self.max_depth:
            return True

        # Not enough samples to split
        if y.size < self.min_samples_split:
            return True

        return False

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """Recursively build the decision tree.

        Parameters
        ----------
        X
            Feature matrix reaching this node.
        y
            Labels reaching this node.
        depth
            Current depth.

        Returns
        -------
        node
            Built node (leaf or internal).
        """
        prediction = self._majority_class(y)
        impurity = self._entropy(y)
        counts = self._class_counts(y)

        # Create a leaf if stopping criteria are met.
        if self._should_stop(y, depth):
            return _Node(
                is_leaf=True,
                prediction=prediction,
                n_samples=int(y.size),
                class_counts=counts,
                impurity=float(impurity),
                info_gain=0.0,
            )

        feature_indices = self._choose_feature_indices()
        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)

        # If no useful split found, make leaf.
        if best_feature is None or best_threshold is None or best_gain < self.min_impurity_decrease:
            return _Node(
                is_leaf=True,
                prediction=prediction,
                n_samples=int(y.size),
                class_counts=counts,
                impurity=float(impurity),
                info_gain=0.0,
            )

        # Split data.
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(
            is_leaf=False,
            prediction=prediction,
            feature_index=int(best_feature),
            threshold=float(best_threshold),
            left=left,
            right=right,
            n_samples=int(y.size),
            class_counts=counts,
            impurity=float(impurity),
            info_gain=float(best_gain),
        )

    def _accumulate_importances(self, node: _Node, importances: np.ndarray) -> None:
        """Accumulate feature importances via information gain.

        Parameters
        ----------
        node
            Current node.
        importances
            Array of shape (n_features_in_,) updated in-place.
        """
        if node.is_leaf:
            return

        assert node.feature_index is not None
        importances[node.feature_index] += node.info_gain

        assert node.left is not None and node.right is not None
        self._accumulate_importances(node.left, importances)
        self._accumulate_importances(node.right, importances)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "decision_tree":
        """Fit the decision tree classifier.

        Parameters
        ----------
        X
            Numeric feature matrix of shape (n_samples, n_features).
        y
            Numeric class labels of shape (n_samples,). Labels are treated as categorical values.

        Returns
        -------
        self
            Fitted estimator.

        Raises
        ------
        TypeError
            If inputs are not numpy arrays or are not numeric.
        ValueError
            If shapes are invalid or arrays contain NaN/inf.
        """
        X_arr = self._check_numeric_matrix(X, name="X")
        y_arr = self._check_numeric_vector(y, name="y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X and y have inconsistent lengths: {X_arr.shape[0]} != {y_arr.shape[0]}")

        if X_arr.shape[0] == 0:
            raise ValueError("Cannot fit on empty dataset.")

        self.n_features_in_ = int(X_arr.shape[1])

        if self.n_features is not None and self.n_features > self.n_features_in_:
            raise ValueError("n_features cannot be greater than the number of input features.")

        self.classes_ = np.unique(y_arr)
        self.root_ = self._build_tree(X_arr, y_arr, depth=0)

        # Compute feature importances.
        importances = np.zeros(self.n_features_in_, dtype=np.float64)
        self._accumulate_importances(self.root_, importances)

        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

        return self

    def _predict_one(self, x: np.ndarray) -> float:
        """Predict the class label for a single sample.

        Parameters
        ----------
        x
            1D feature vector of shape (n_features_in_,).

        Returns
        -------
        y_hat
            Predicted class label.
        """
        if self.root_ is None:
            raise RuntimeError("This decision_tree instance is not fitted yet. Call fit(X, y) first.")

        node = self.root_
        while not node.is_leaf:
            assert node.feature_index is not None and node.threshold is not None
            if x[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return float(node.prediction)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X
            Numeric feature matrix of shape (n_samples, n_features).

        Returns
        -------
        y_pred
            Predicted labels of shape (n_samples,).
        """
        X_arr = self._check_numeric_matrix(X, name="X")

        if self.n_features_in_ is None or self.root_ is None:
            raise RuntimeError("This decision_tree instance is not fitted yet. Call fit(X, y) first.")

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but this tree was fit with {self.n_features_in_} features."
            )

        return np.array([self._predict_one(row) for row in X_arr], dtype=np.float64)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities for samples in X.

        This estimates probabilities using the class distribution at the reached leaf.

        Parameters
        ----------
        X
            Numeric feature matrix of shape (n_samples, n_features).

        Returns
        -------
        proba
            Array of shape (n_samples, n_classes) with rows summing to 1.
        """
        X_arr = self._check_numeric_matrix(X, name="X")

        if self.n_features_in_ is None or self.root_ is None or self.classes_ is None:
            raise RuntimeError("This decision_tree instance is not fitted yet. Call fit(X, y) first.")

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but this tree was fit with {self.n_features_in_} features."
            )

        class_to_index = {float(c): i for i, c in enumerate(self.classes_)}
        out = np.zeros((X_arr.shape[0], self.classes_.size), dtype=np.float64)

        for i, row in enumerate(X_arr):
            node = self.root_
            while not node.is_leaf:
                assert node.feature_index is not None and node.threshold is not None
                if row[node.feature_index] <= node.threshold:
                    assert node.left is not None
                    node = node.left
                else:
                    assert node.right is not None
                    node = node.right

            counts = node.class_counts or {}
            total = sum(counts.values())
            if total == 0:
                # Fallback (shouldn't happen): uniform distribution
                out[i, :] = 1.0 / self.classes_.size
            else:
                for cls, cnt in counts.items():
                    out[i, class_to_index[float(cls)]] = cnt / total

        return out

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Compute mean accuracy on the given test data and labels.

        Parameters
        ----------
        X
            Numeric feature matrix of shape (n_samples, n_features).
        y
            Numeric class labels of shape (n_samples,).

        Returns
        -------
        accuracy
            Mean accuracy.
        """
        y_true = self._check_numeric_vector(y, name="y")
        y_pred = self.predict(X)

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("X and y have inconsistent lengths.")

        return float(np.mean(y_pred == y_true))

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Returns
        -------
        params
            Dictionary of constructor parameters.
        """
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "n_features": self.n_features,
            "random_state": self.random_state,
            "min_impurity_decrease": self.min_impurity_decrease,
        }

    def set_params(self, **params: Any) -> "decision_tree":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Estimator parameters.

        Returns
        -------
        self
            Estimator instance.
        """
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k!r} for decision_tree.")
            setattr(self, k, v)

        # Re-validate after mutation
        self.min_impurity_decrease = float(self.min_impurity_decrease)
        self._validate_hyperparameters()
        self._rng = np.random.default_rng(self.random_state)

        return self

    def __repr__(self) -> str:
        """String representation of the estimator.

        Returns
        -------
        rep
            String representation.
        """
        params = self.get_params()
        args = ", ".join(f"{k}={v!r}" for k, v in params.items())
        fitted = self.root_ is not None
        return f"decision_tree({args}){' [fitted]' if fitted else ''}"