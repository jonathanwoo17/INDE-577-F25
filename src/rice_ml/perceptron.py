"""
perceptron.py
--------------

This module implements a simple binary linear Perceptron classifier from scratch 
using NumPy. The Perceptron algorithm is one of the earliest and most fundamental 
supervised learning algorithms for classification. It iteratively updates a weight 
vector to find a linear decision boundary that separates two classes.

The module also includes optional visualization utilities using Matplotlib and 
Seaborn for exploratory analysis and model evaluation.

Dependencies
------------
- numpy
- pandas
- matplotlib
- seaborn
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

class Perceptron:
    """
    A simple implementation of a binary linear perceptron classifier.

    The perceptron learns a linear decision boundary between two classes
    by iteratively updating weights based on misclassified samples.

    Attributes
    ----------
    learning_rate : float
        Step size for weight updates during training.
    iterations : int
        Number of passes (epochs) over the training dataset.
    vector_col : str
        Column name containing feature vectors.
    label_col : str
        Column name containing labels.
    w_ : np.ndarray
        Learned weight vector after training.
    b_ : float
        Learned bias term after training.
    _label_map_ : dict
        Mapping from original labels to numeric {-1, +1}.
    _inv_label_map_ : dict
        Reverse mapping from numeric labels to original labels.
    """

    def __init__(self, learning_rate: float = 1.0, iterations: int = 50,
                 vector_col: str = "x", label_col: str = "y"):
        """
        Initialize the perceptron model with given hyperparameters.

        Parameters
        ----------
        learning_rate : float, optional
            Controls the size of each parameter update (default=1.0).
        iterations : int, optional
            Number of training epochs (default=50).
        vector_col : str, optional
            Name of the DataFrame column containing feature vectors (default="x").
        label_col : str, optional
            Name of the DataFrame column containing class labels (default="y").
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.vector_col = vector_col
        self.label_col = label_col

        # Initialize bias and label mapping dictionaries
        self.b_: float = 0.0
        self._label_map_: dict = {}
        self._inv_label_map_: dict = {}
    
    def prepare_data(self, df):
        """
        Prepare feature and label arrays for training or prediction.

        This converts the DataFrame's vector column into a 2D numpy array
        and maps categorical labels into {-1, +1}.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing at least the feature and label columns.

        Returns
        -------
        X : np.ndarray
            2D array of feature vectors.
        y : np.ndarray
            1D array of numeric labels (-1 or +1).
        """

        X = np.vstack(df[self.vector_col].apply(np.asarray).to_numpy())
        y_raw = df[self.label_col].to_numpy()

        # Ensure binary classification
        categories = np.unique(y_raw)
        if len(categories) != 2:
            raise ValueError("Perceptron only supports binary classification.")

        # Map categories to binary {-1, 1}
        self._label_map_ = {categories[0]: -1, categories[1]: +1}
        self._inv_label_map_ = {-1: categories[0], +1: categories[1]}

        y = np.vectorize(self._label_map_.get)(y_raw)
        return X, y

    def pre_activation(self, X):
        """
        Compute the linear combination of inputs and weights (Xw + b).

        Parameters
        ----------
        X : np.ndarray
            2D array of input feature vectors.

        Returns
        -------
        np.ndarray
            The pre-activation (margin) values for each sample.
        """
        return np.dot(X, self.w_) + self.b_

    def train(self, df):
        """
        Train the perceptron model on the provided dataset.

        The algorithm updates weights for each misclassified sample.
        Training continues for the specified number of iterations.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataset containing feature vectors and labels.

        Returns
        -------
        self : Perceptron
            The trained perceptron instance.
        """
        X, y = self.prepare_data(df)
        n_features = X.shape[1]

        # Initialize weights and bias with random values
        self.w_ = np.random.normal(0, 1e-6, n_features)
        self.b_ = np.random.normal(0, 1e-6)

        # Loop over dataset multiple epochs
        for _ in range(self.iterations):
            for xi, yi in zip(X, y):
                margin = np.dot(self.w_, xi) + self.b_
                if yi * margin <= 0:
                    self.w_ += self.learning_rate * yi * xi
                    self.b_ += self.learning_rate * yi
        return self

        
    def predict(self, df):
        """
        Predict class labels for new data points.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing feature vectors to classify.

        Returns
        -------
        np.ndarray
            Predicted labels in the original label format.
        """
        X = np.vstack(df[self.vector_col].apply(np.asarray).to_numpy())
        margins = self.pre_activation(X)

        # Convert margins into predicted {-1, +1} labels
        y_pm = np.where(margins >= 0, +1, -1)

        # Map back to original labels
        return np.vectorize(self._inv_label_map_.get)(y_pm)

    def score(self, df: pd.DataFrame):
        """
        Compute the model's accuracy on a given dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset with true labels for evaluation.

        Returns
        -------
        float
            Accuracy score (fraction of correctly predicted labels).
        """
        y_true = df[self.label_col].to_numpy()
        y_pred = self.predict(df)
        return (y_true == y_pred).mean()

    def loss(self, df):
        """
        Compute a simple squared error loss for evaluation.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing feature vectors and true labels.

        Returns
        -------
        float
            Total squared error loss.
        """
        X = np.vstack(df[self.vector_col].apply(np.asarray).to_numpy())
        y_true = np.vectorize(self._label_map_.get)(df[self.label_col].to_numpy())
        y_hat = np.where((X @ self.w_ + self.b_) >= 0, 1, -1)

        # Loss = 0.25 * sum((y_hat - y_true)^2); AKA counting misclassifications
        return 0.25 * np.sum((y_hat - y_true) ** 2)
