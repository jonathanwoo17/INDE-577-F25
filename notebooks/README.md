# Jupyter Notebooks

This directory contains interactive Jupyter notebooks that demonstrate and visualize
the functionality of the algorithms implemented in the `src/rice_ml` package.  
Each notebook is designed to show both how the algorithm is implemented and why
it works from a mathematical and conceptual standpoint.

---

## Purpose

These notebooks serve as an educational sandbox for:

- Testing and visualizing algorithms from the `src/` package  
- Exploring mathematical intuition behind classic ML models  
- Demonstrating small-scale experiments and result interpretation  
- Building a bridge between *theory*, *implementation*, and *practice*

---

## Contents

The following algorithms are currently included"

- The Perceptron
- xxx

---

## Overview of Included Algorithms

### Perceptron

The Perceptron is one of the earliest linear classifiers, introduced by Frank Rosenblatt in 1958. It learns a hyperplane that separates two classes using iterative weight updates. For each data point \( (x_i, y_i) \) with \( y_i \in \{-1, +1\} \), predictions are made as:

\[
\hat{y}_i = \text{sign}(w^T x_i + b)
\]

If a sample is misclassified (\( y_i (w^T x_i + b) \le 0 \)), the model updates:

\[
w \leftarrow w + \eta y_i x_i, \quad b \leftarrow b + \eta y_i
\]

where \( \eta \) is the learning rate. This process shifts the decision boundary toward misclassified samples until convergence.

The Perceptron works well for linearly separable data and is often used to introduce the fundamentals of supervised learning and neural network concepts.