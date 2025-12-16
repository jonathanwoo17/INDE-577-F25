"""Ensemble methods for supervised learning.

This module implements a simple bagging classifier that combines multiple
``decision_tree`` base learners to reduce variance. The implementation uses
 components from ``rice_ml`` and numpy.

Dependencies:
- numpy
- rice_ml.decision_tree
- typing
"""


from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .decision_tree import decision_tree


class bagging_classifier:
    """Bootstrap aggregating (bagging) ensemble for classification.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of base ``decision_tree`` classifiers to train. Must be at least 1.
    max_samples : float, default=1.0
        Fraction of the training set to sample (with replacement) for each
        estimator. Must be in the range (0, 1].
    random_state : int or None, default=None
        Optional random seed for reproducible bootstrapping.
    base_params : dict or None, default=None
        Optional dictionary of hyperparameters passed to each ``decision_tree``
        base learner (e.g., ``{"max_depth": 5, "n_features": 3}``).

    Attributes
    ----------
    estimators_ : list of decision_tree
        List of fitted ``decision_tree`` instances.
    classes_ : ndarray or None
        Sorted unique class labels observed during fitting.
    n_features_in_ : int or None
        Number of features seen during fitting.
    _rng : numpy.random.Generator
        Random number generator used for bootstrapping.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        random_state: Optional[int] = None,
        base_params: Optional[Dict[str, float]] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = float(max_samples)
        self.random_state = random_state
        self.base_params = base_params or {}

        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.estimators_: List[decision_tree] = []

        self._rng: np.random.Generator = np.random.default_rng(random_state)
        self._validate_hyperparameters()

    def _validate_hyperparameters(self) -> None:
        """Validate constructor arguments for type and value.

        Raises
        ------
        TypeError
            If ``n_estimators`` is not an integer, ``max_samples`` is not numeric,
            ``random_state`` is not ``None`` or an integer, or ``base_params`` is
            not a dictionary.
        ValueError
            If ``n_estimators`` is less than 1 or ``max_samples`` is outside the
            interval (0, 1].
        """
        if not isinstance(self.n_estimators, int):
            raise TypeError("n_estimators must be an integer.")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be an integer >= 1.")

        if not isinstance(self.max_samples, (int, float)):
            raise TypeError("max_samples must be a float in the interval (0, 1].")
        if not (0 < self.max_samples <= 1.0):
            raise ValueError("max_samples must be a float in the interval (0, 1].")

        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError("random_state must be None or an integer.")

        if not isinstance(self.base_params, dict):
            raise TypeError("base_params must be a dictionary of decision_tree keyword arguments.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "bagging_classifier":
        """Fit the bagging ensemble on the provided training data.

        Parameters
        ----------
        X : ndarray
            Feature matrix of shape ``(n_samples, n_features)``. Must be numeric
            and finite.
        y : ndarray
            Label vector of shape ``(n_samples,)``. Must be numeric and finite.

        Returns
        -------
        bagging_classifier
            The fitted estimator.

        Raises
        ------
        TypeError
            If ``X`` or ``y`` is not a numpy array or contains non-numeric data.
        ValueError
            If ``X`` or ``y`` has an invalid shape, contains NaN/inf, or if
            there is a mismatch between the number of samples in ``X`` and ``y``.
        RuntimeError
            If fitting fails due to an unexpected error in base estimators.
        """
        X_arr = decision_tree._check_numeric_matrix(X, name="X")
        y_arr = decision_tree._check_numeric_vector(y, name="y")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                "X and y must have the same number of samples; "
                f"got n_samples={X_arr.shape[0]} and n_labels={y_arr.shape[0]}."
            )

        n_samples, n_features = X_arr.shape
        n_bootstrap = max(1, int(np.floor(self.max_samples * n_samples)))

        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = n_features
        self.estimators_ = []

        for _ in range(self.n_estimators):
            indices = self._rng.choice(n_samples, size=n_bootstrap, replace=True)
            estimator = decision_tree(**self.base_params)
            estimator.fit(X_arr[indices], y_arr[indices])
            self.estimators_.append(estimator)

        return self

    def _check_is_fitted(self) -> None:
        """Ensure the estimator has been fitted before inference.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        """
        if not self.estimators_:
            raise RuntimeError("This bagging_classifier is not fitted yet. Call 'fit' first.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities by averaging over base estimators.

        Parameters
        ----------
        X : ndarray
            Feature matrix of shape ``(n_samples, n_features)`` with numeric,
            finite values.

        Returns
        -------
        ndarray
            Array of shape ``(n_samples, n_classes)`` containing averaged class
            probabilities.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        TypeError
            If ``X`` is not a numeric numpy array.
        ValueError
            If ``X`` has an invalid shape or contains NaN/inf values.
        """
        self._check_is_fitted()
        X_arr = decision_tree._check_numeric_matrix(X, name="X")

        if self.n_features_in_ is not None and X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "Number of features in X does not match training data: "
                f"expected {self.n_features_in_}, got {X_arr.shape[1]}."
            )

        probas = [est.predict_proba(X_arr) for est in self.estimators_]
        return np.mean(probas, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using majority vote across base estimators.

        Parameters
        ----------
        X : ndarray
            Feature matrix of shape ``(n_samples, n_features)`` with numeric,
            finite values.

        Returns
        -------
        ndarray
            Array of shape ``(n_samples,)`` containing predicted class labels.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted or class labels are missing.
        TypeError
            If ``X`` is not a numeric numpy array.
        ValueError
            If ``X`` has an invalid shape or contains NaN/inf values.
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        if self.classes_ is None:
            raise RuntimeError("Classes are undefined; ensure the model has been fitted.")
        return self.classes_[class_indices]

    def feature_importances(self) -> np.ndarray:
        """Average feature importances across all base estimators.

        Returns
        -------
        ndarray
            Array of shape ``(n_features,)`` representing averaged feature
            importances across all fitted base estimators.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted or base estimators do not
            expose ``feature_importances_``.
        """
        self._check_is_fitted()
        importances = [
            est.feature_importances_ for est in self.estimators_ if est.feature_importances_ is not None
        ]
        if not importances:
            raise RuntimeError("Base estimators do not expose feature_importances_.")
        return np.mean(importances, axis=0)

    def __repr__(self) -> str:
        return (
            f"bagging_classifier(n_estimators={self.n_estimators}, "
            f"max_samples={self.max_samples}, base_params={self.base_params})"
        )