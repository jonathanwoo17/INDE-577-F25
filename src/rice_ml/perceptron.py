import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

class Perceptron:
    """
    """

    def __init__(self, learning_rate: float = 1.0, iterations: int = 50,
                 vector_col: str = "x", label_col: str = "y"):
        """
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.vector_col = vector_col
        self.label_col = label_col

        self.b_: float = 0.0
        self._label_map_: dict = {}
        self._inv_label_map_: dict = {}
    
    def prepare_data(self, df):
        """
        """

        X = np.vstack(df[self.vector_col].apply(np.asarray).to_numpy())
        y_raw = df[self.label_col].to_numpy()

        categories = np.unique(y_raw)
        if len(categories) != 2:
            raise ValueError("Perceptron only supports binary classification.")

        # Map categories
        self._label_map_ = {categories[0]: -1, categories[1]: +1}
        self._inv_label_map_ = {-1: categories[0], +1: categories[1]}

        y = np.vectorize(self._label_map_.get)(y_raw)
        return X, y

    def pre_activation(self, X):
        """
        """
        return np.dot(X, self.w_) + self.b_

    def train(self, df):
        """
        """
        X, y = self.prepare_data(df)
        n_features = X.shape[1]

        self.w_ = np.random.normal(0, 1e-6, n_features)
        self.b_ = np.random.normal(0, 1e-6)

        for _ in range(self.iterations):
            for xi, yi in zip(X, y):
                margin = np.dot(self.w_, xi) + self.b_
                if yi * margin <= 0:
                    self.w_ += self.learning_rate * yi * xi
                    self.b_ += self.learning_rate * yi
        return self

        
    def predict(self, df):
        """
        """
        X = np.vstack(df[self.vector_col].apply(np.asarray).to_numpy())
        margins = self.pre_activation(X)
        y_pm = np.where(margins >= 0, +1, -1)

        return np.vectorize(self._inv_label_map_.get)(y_pm)

    def score(self, df: pd.DataFrame):
        """
        """
        y_true = df[self.label_col].to_numpy()
        y_pred = self.predict(df)
        return (y_true == y_pred).mean()

    def loss(self, df):
        """
        """
        X = np.vstack(df[self.vector_col].apply(np.asarray).to_numpy())
        y_true = np.vectorize(self._label_map_.get)(df[self.label_col].to_numpy())
        y_hat = np.where((X @ self.w_ + self.b_) >= 0, 1, -1)
        return 0.25 * np.sum((y_hat - y_true) ** 2)
