"""
Principal Component Analysis (scikit-learn backend).

This module provides a simple PCA implementation with a NumPy-style API,
internally using scikit-learn for numerical robustness.

Dependencies
------------
- numpy
- sklearn
- typing
"""

from __future__ import annotations
from typing import Optional, Union, Sequence

import numpy as np
from sklearn.decomposition import PCA as _SklearnPCA
from sklearn.preprocessing import StandardScaler

__all__ = ["pca"]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Ensure input is a non-empty 2D NumPy array of floats.

    Parameters
    ----------
    X : array_like
        Input data.
    name : str, default="X"
        Name used in error messages.

    Returns
    -------
    ndarray of shape (n_samples, n_features)
        Validated array.

    Raises
    ------
    ValueError
        If input is empty or has invalid dimensionality.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D array, got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


class pca:
    """
    Principal Component Analysis (PCA).

    Linear dimensionality reduction using an orthogonal transformation
    to project data into a lower-dimensional space.

    Parameters
    ----------
    n_components : int or float or None, default=None
        Number of components to keep:
        - int: exact number of components
        - float in (0, 1): fraction of variance to explain
        - None: keep all components
    scale : bool, default=True
        Whether to standardize features before applying PCA.

    Attributes
    ----------
    n_components_ : int
        Number of components retained after fitting.
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space.
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each principal component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total variance explained by each component.
    singular_values_ : ndarray of shape (n_components,)
        Singular values corresponding to each component.
    mean_ : ndarray of shape (n_features,)
        Per-feature mean of the training data.
    n_features_ : int
        Number of features seen during fitting.
    n_samples_ : int
        Number of samples seen during fitting.
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        scale: bool = True,
    ) -> None:
        """
        Initialize the PCA model.

        Parameters
        ----------
        n_components : int or float or None, default=None
            Number of components to retain.
        scale : bool, default=True
            Whether to standardize input features.
        """
        self.n_components = n_components
        self.scale = scale

        self.n_components_: Optional[int] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None

        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[_SklearnPCA] = None

    def fit(self, X: ArrayLike) -> "pca":
        """
        Fit the PCA model to the data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : PCA
            Fitted PCA instance.
        """
        X_arr = _ensure_2d_float(X, "X")
        self.n_samples_, self.n_features_ = X_arr.shape

        if self.scale:
            self._scaler = StandardScaler()
            X_arr = self._scaler.fit_transform(X_arr)
            self.mean_ = self._scaler.mean_
        else:
            self.mean_ = np.mean(X_arr, axis=0)

        self._pca = _SklearnPCA(n_components=self.n_components)
        self._pca.fit(X_arr)

        self.n_components_ = self._pca.n_components_
        self.components_ = self._pca.components_
        self.explained_variance_ = self._pca.explained_variance_
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self.singular_values_ = self._pca.singular_values_

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Apply dimensionality reduction to the data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self._check_fitted()
        X_arr = _ensure_2d_float(X, "X")

        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_}."
            )

        if self.scale:
            X_arr = self._scaler.transform(X_arr)

        return self._pca.transform(X_arr)

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Fit PCA to data, then apply the transformation.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform data back to the original feature space.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_components)
            Data in reduced space.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        self._check_fitted()
        X_arr = _ensure_2d_float(X, "X")

        if X_arr.shape[1] != self.n_components_:
            raise ValueError(
                f"X has {X_arr.shape[1]} components, expected {self.n_components_}."
            )

        X_rec = self._pca.inverse_transform(X_arr)

        if self.scale:
            X_rec = self._scaler.inverse_transform(X_rec)

        return X_rec

    def score(self, X: ArrayLike) -> float:
        """
        Compute a reconstruction-based score.

        The score is the negative mean squared reconstruction error.
        Higher values indicate better reconstruction.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        float
            Reconstruction score.
        """
        self._check_fitted()
        X_arr = _ensure_2d_float(X, "X")

        Z = self.transform(X_arr)
        X_hat = self.inverse_transform(Z)

        return -np.mean((X_arr - X_hat) ** 2)

    def _check_fitted(self) -> None:
        """
        Check whether the PCA model has been fitted.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        if self._pca is None:
            raise RuntimeError("PCA instance is not fitted yet.")
