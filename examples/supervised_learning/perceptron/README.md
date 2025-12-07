# The Perceptron

This directory contains example code and notes for the Perceptron algorithm
in supervised learning.

---

## Algorithm

The Perceptron is one of the earliest linear classifiers, introduced by Frank Rosenblatt in 1958. It learns a hyperplane that separates two classes using iterative weight updates. For each data point (xᵢ, yᵢ) with yᵢ ∈ {−1, +1}, predictions are made as:

**ŷᵢ = sign(wᵀxᵢ + b)**

If a sample is misclassified (yᵢ(wᵀxᵢ + b) ≤ 0), the model updates:

**w ← w + ηyᵢxᵢ**, **b ← b + ηyᵢ**

where η is the learning rate. This process shifts the decision boundary toward misclassified samples until convergence.

The Perceptron works well for linearly separable data and is often used to introduce the fundamentals of supervised learning and neural network concepts.
