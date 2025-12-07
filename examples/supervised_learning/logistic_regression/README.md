# Logistic Regression

This directory contains example code and notes for the Logistic Regression algorithm
in supervised learning.

---

## Algorithm

Logistic Regression is a linear classifier that models the **probability** that an input
belongs to the positive class. Instead of applying a hard sign to the linear score,
it passes the score through the **sigmoid** function to obtain a value in \([0, 1]\).

For each data point \((xᵢ, yᵢ)\) with \(yᵢ ∈ \{0, 1\}\), we compute:

**zᵢ = wᵀxᵢ + b**  
**pᵢ = σ(zᵢ) = 1 / (1 + exp(−zᵢ))**

Here, \(pᵢ\) is interpreted as the model’s estimate of \(P(yᵢ = 1 \mid xᵢ)\).

To learn the parameters \(w\) and \(b\), Logistic Regression minimizes the
**binary cross-entropy (logistic) loss**:

**L(w, b) = − (1 / n) Σ [ yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ ) ]**

Gradient descent is used to update the parameters:

**w ← w − η ∂L/∂w**, **b ← b − η ∂L/∂b**

where η is the learning rate.

---

### Comparison to the Perceptron

Unlike the Perceptron, which only outputs a hard class label based on the sign of
\(wᵀx + b\) and updates weights only on misclassified examples.

Logistic Regression:

- Optimizes a **smooth, well-defined loss function**
- Produces **probability estimates** rather than just yes or no classifications
- Uses gradient-based updates that depend on how confidently the model is making
  each prediction.
