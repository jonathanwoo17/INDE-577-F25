import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

class Perceptron:
    """
    
    """

    def __init___(self, learning_rate: float = 1.0, iterations: int = 50,
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
    
    def train(self, X, y):
        
    def predict(self, x):



def add(a, b):
    return a + b