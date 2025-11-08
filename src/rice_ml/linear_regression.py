"""
linear_regression.py
--------------------

This module implements a simple single-neuron linear regression model from scratch 
using NumPy. The model learns to approximate a linear relationship between input 
features and a continuous target variable through stochastic gradient descent (SGD).

The implementation provides methods for training, prediction, and data preparation 
from pandas DataFrames. The class is designed for educational and experimental 
purposes to illustrate the mechanics of linear regression and gradient-based learning 
without relying on external machine learning libraries.

Dependencies
------------
- numpy
"""

import numpy as np

class SingleNeuron:
    """
    A simple single-neuron linear regression model trained using stochastic gradient descent (SGD).
    
    Parameters
    ----------
    activation_function : function
        Function applied to the linear combination of inputs (default is identity).

    Attributes
    ----------
    w_ : np.ndarray
        Weight vector (including bias as the last element).
    errors_ : list
        List of mean squared errors per epoch.
    """

    def __init__(self, activation_function=lambda x: x):
        self.activation_function = activation_function

        self.vector_col = None
        self.target_col = None

    def train(self, X, y, alpha=0.01, epochs=100, random_state=None):
        """
        Train the neuron using stochastic gradient descent (SGD).
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features.
        y : np.ndarray, shape (n_samples,)
            Target values.
        alpha : float
            Learning rate.
        epochs : int
            Number of passes through the dataset.
        random_state : int or None
            Optional seed for reproducibility.
        """
        if random_state is not None:
            np.random.seed(random_state)

        # initialize weights (including bias)
        self.w_ = np.random.randn(X.shape[1] + 1)
        self.errors_ = []
        N = X.shape[0]

        for _ in range(epochs):
            total_error = 0
            for xi, target in zip(X, y):
                # Forward pass
                output = self.predict(xi)
                error = output - target

                # Weight update
                self.w_[:-1] -= alpha * error * xi
                self.w_[-1] -= alpha * error

                total_error += 0.5 * (error ** 2)
            self.errors_.append(total_error / N)
        return self

    def predict(self, X):
        """
        Make predictions on input data X.
        """
        X = np.asarray(X)
        z = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return self.activation_function(z)
    
    def prepare_data(self, df):
        """
        Prepare feature and target arrays for linear regression.

        Converts the DataFrame's feature (vector) column into a 2D NumPy array
        and ensures that the target column contains numeric values.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing at least the feature and target columns.

        Returns
        -------
        X : np.ndarray
            2D array of feature vectors, shape (n_samples, n_features).
        y : np.ndarray
            1D array of numeric target values, shape (n_samples,).
        """
        # Convert vector column to 2D NumPy array
        X = np.vstack(df[self.vector_col].apply(np.asarray).to_numpy())

        # Extract target values
        y = df[self.target_col].to_numpy()

        # Validate numeric targets
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("Linear regression requires numeric target values.")

        return X, y
        

    

