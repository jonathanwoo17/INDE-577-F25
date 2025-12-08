"""
logistic_regression.py
----------------------

Implementation of a simple binary Logistic Regression classifier.

Logistic regression models the probability that a given input belongs
to the positive class using a sigmoid applied to a linear combination
of features. The parameters are learned by minimizing the binary
cross-entropy (logistic) loss via gradient descent.

This implementation is designed for educational purposes and mirrors
the DataFrame-based interface used in the Perceptron module.

Dependencies
------------
- numpy
- pandas
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class LogisticRegression:
    """
    Binary logistic regression classifier trained with gradient descent.

    The model assumes a linear decision function of the form

        z = x^T w + b

    and converts it to a probability via the sigmoid function

        p(y = 1 | x) = 1 / (1 + exp(-z)).

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent updates.
    n_epochs : int, default=100
        Number of passes (epochs) over the training data.
    vector_col : str, default="x"
        Name of the DataFrame column containing feature vectors.
    label_col : str, default="y"
        Name of the DataFrame column containing binary labels.
    random_state : int or None, optional
        Seed for weight initialization. If None, NumPy's global RNG is used.

    Attributes
    ----------
    w_ : np.ndarray of shape (n_features,)
        Learned weight vector after training.
    b_ : float
        Learned bias term after training.
    losses_ : list[float]
        Training loss (binary cross-entropy) recorded at each epoch.
    _label_map_ : dict
        Mapping from original labels to numeric {0, 1}.
    _inv_label_map_ : dict
        Reverse mapping from numeric labels {0, 1} back to original labels.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        vector_col: str = "x",
        label_col: str = "y",
        random_state: Optional[int] = None,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.vector_col = vector_col
        self.label_col = label_col
        self.random_state = random_state

        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.losses_: list[float] = []

        self._label_map_: Dict[object, int] = {}
        self._inv_label_map_: Dict[int, object] = {}

    # ------------------------------------------------------------------
    # Data preparation / utilities
    # ------------------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature and label arrays for training or prediction.

        This converts the DataFrame's vector column into a 2D NumPy array
        and maps arbitrary binary labels into {0, 1}.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing at least the feature and label columns.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            2D array of feature vectors.
        y : np.ndarray of shape (n_samples,)
            1D array of numeric labels in {0, 1}.

        Raises
        ------
        ValueError
            If the number of unique labels is not exactly 2.
        """
        X = np.vstack(df[self.vector_col].apply(np.asarray).to_numpy())
        y_raw = df[self.label_col].to_numpy()

        categories = np.unique(y_raw)
        if len(categories) != 2:
            raise ValueError("LogisticRegression only supports binary classification.")

        # Map categories to {0, 1}
        self._label_map_ = {categories[0]: 0, categories[1]: 1}
        self._inv_label_map_ = {0: categories[0], 1: categories[1]}

        y = np.vectorize(self._label_map_.get)(y_raw)
        y = y.astype(float)
        return X, y

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Compute the element-wise sigmoid function.

        Parameters
        ----------
        z : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Array with sigmoid applied element-wise.
        """
        # Numerically stable sigmoid
        z = np.asarray(z, dtype=float)
        return 1.0 / (1.0 + np.exp(-z))

    def pre_activation(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the linear combination of inputs and weights (Xw + b).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Pre-activation values for each sample.
        """
        if self.w is None:
            raise RuntimeError("Model is not trained. Call `fit` first.")
        return X @ self.w + self.b

    # Properties to avoid accidental masking by type checker
    @property
    def w(self) -> np.ndarray:
        """Weight vector (internal helper property)."""
        if self.w_ is None:
            raise RuntimeError("Weights not initialized. Call `fit` first.")
        return self.w_

    @property
    def b(self) -> float:
        """Bias term (internal helper property)."""
        return self.b_

    # ------------------------------------------------------------------
    # Core API: fit / predict / evaluate
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "LogisticRegression":
        """
        Fit the logistic regression model on the provided dataset.

        The parameters are optimized using batch gradient descent on the
        binary cross-entropy (logistic) loss.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataset containing feature vectors and labels.

        Returns
        -------
        self : LogisticRegression
            The trained logistic regression instance.
        """
        X, y = self.prepare_data(df)
        n_samples, n_features = X.shape

        # Initialize parameters
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(0.0, 1e-3, size=n_features)
        self.b_ = float(rng.normal(0.0, 1e-3))

        self.losses_ = []
        lr = self.learning_rate

        for _ in range(self.n_epochs):
            # Forward pass
            logits = X @ self.w_ + self.b_
            probs = self._sigmoid(logits)  # p(y=1 | x)

            # Compute binary cross-entropy loss
            eps = 1e-12
            loss = -np.mean(
                y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps)
            )
            self.losses_.append(loss)

            # Gradients (batch gradient descent)
            error = probs - y  # shape (n_samples,)
            grad_w = X.T @ error / n_samples
            grad_b = np.mean(error)

            # Parameter update
            self.w_ -= lr * grad_w
            self.b_ -= lr * grad_b

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for the positive class on new data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing feature vectors to evaluate.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted probabilities p(y=1 | x).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not trained. Call `fit` first.")

        X = np.vstack(df[self.vector_col].apply(np.asarray).to_numpy())
        logits = X @ self.w_ + self.b_
        return self._sigmoid(logits)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for new data points.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing feature vectors to classify.
        threshold : float, default=0.5
            Threshold applied to the predicted probability. Values
            >= threshold are assigned to the positive class.

        Returns
        -------
        np.ndarray
            Predicted labels in the original label format.
        """
        probs = self.predict_proba(df)
        y_num = (probs >= threshold).astype(int)
        return np.vectorize(self._inv_label_map_.get)(y_num)

    def score(self, df: pd.DataFrame, threshold: float = 0.5) -> float:
        """
        Compute the model's accuracy on a given dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset with true labels for evaluation.
        threshold : float, default=0.5
            Threshold used to convert probabilities into class labels.

        Returns
        -------
        float
            Accuracy score (fraction of correctly predicted labels).
        """
        y_true = df[self.label_col].to_numpy()
        y_pred = self.predict(df, threshold=threshold)
        return (y_true == y_pred).mean()

    def loss(self, df: pd.DataFrame) -> float:
        """
        Compute the binary cross-entropy loss on a dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing feature vectors and true labels.

        Returns
        -------
        float
            Average binary cross-entropy loss.
        """
        if self.w_ is None:
            raise RuntimeError("Model is not trained. Call `fit` first.")

        X = np.vstack(df[self.vector_col].apply(np.asarray).to_numpy())
        y_raw = df[self.label_col].to_numpy()
        y = np.vectorize(self._label_map_.get)(y_raw).astype(float)

        logits = X @ self.w_ + self.b_
        probs = self._sigmoid(logits)

        eps = 1e-12
        loss = -np.mean(
            y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps)
        )
        return loss