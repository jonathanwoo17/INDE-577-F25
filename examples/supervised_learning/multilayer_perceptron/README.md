# Multilayer Perceptron (Neural Network)

This directory contains example code and notes for the Multilayer Perceptron algorithm
in supervised learning.

---

## Algorithm

A Multilayer Perceptron (MLP) extends the basic perceptron by stacking multiple layers of neurons
with nonlinear activation functions. This allows the model to learn complex, nonlinear decision
boundaries.

For an input vector **x**, a simple MLP with one hidden layer computes:

- **Hidden layer:**  
  **h = σ(W₁x + b₁)**
- **Output layer (for binary classification):**  
  **z = W₂h + b₂**  
  **ŷ = σ(z)** or **ŷ = softmax(z)** for multi-class problems  

Here, **W₁, W₂** are weight matrices, **b₁, b₂** are bias vectors, and **σ(·)** is a nonlinear
activation (e.g., ReLU, tanh, or sigmoid).

The network is trained by minimizing a loss function (e.g., cross-entropy) over a dataset
{(xᵢ, yᵢ)}. Training proceeds via:

1. **Forward pass:** compute predictions ŷᵢ for each xᵢ.  
2. **Loss computation:** measure the discrepancy between ŷᵢ and yᵢ.  
3. **Backward pass (backpropagation):** compute gradients of the loss with respect to all
   weights and biases.  
4. **Parameter update:** adjust parameters using gradient-based methods, e.g.:  
   **θ ← θ − η ∇θ L(θ)**  

where **θ** collects all weights and biases, **η** is the learning rate, and **L** is the loss.

Because of the layered structure and nonlinear activations, MLPs can approximate a wide range of
functions and solve problems where classes are not necessarily linearly separable.

---

## Data