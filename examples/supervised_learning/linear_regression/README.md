# Linear Regression and Stochastic Gradient Descent

This directory contains example code and notes for the Linear Regression algorithm
in supervised learning.

---

## Algorithm

Linear Regression aims to find the best-fitting line that minimizes the difference between
predicted and actual values in continuous output data. Unlike the Perceptron, which performs
classification, linear regression predicts a real-valued output \hat{y} given an input vector x
by modeling a linear relationship:

\hat{y}_i = w^T x_i + b

The objective is to minimize the Mean Squared Error (MSE) between predictions and actual targets:

L(w, b) = (1 / n) * sum_{i=1}^n (y_i - \hat{y}_i)^2

Using stochastic gradient descent (SGD), parameters are updated incrementally for each sample to
reduce the loss:

w := w - \eta * dL/dw,   b := b - \eta * dL/db

For a single sample (x_i, y_i), the partial derivatives are:

dL/dw = -2 * x_i * (y_i - \hat{y}_i)
dL/db = -2 * (y_i - \hat{y}_i)

where \eta represents the learning rate. This iterative update rule gradually adjusts the model
parameters to minimize prediction errors across the dataset.

Linear regression with SGD is widely used due to its simplicity, scalability to large datasets,
and interpretability, making it a foundational method in both machine learning and statistics.

---

## Data