# K-Nearest Neighbors

This directory contains example code and notes for the K Nearest Neighbors algorithm
in supervised learning.

---

## Algorithm

K-Nearest Neighbors (k-NN) is a non-parametric, instance-based learning algorithm used for both classification and regression. Unlike linear models that learn explicit parameters (w, b), k-NN makes predictions by comparing a new input xᵢ to the training data and using the labels of the “nearest” examples.

Given a query point xᵢ, k-NN identifies the set Nₖ(xᵢ) containing the k closest training samples according to a distance metric, typically Euclidean distance:

d(xᵢ, xⱼ) = ‖xᵢ − xⱼ‖₂

For classification, the predicted label ŷᵢ is obtained by majority vote among the neighbors:

ŷᵢ = mode{ yⱼ ∣ xⱼ ∈ Nₖ(xᵢ) }

For regression, the prediction is often the average of the neighbors’ targets:

ŷᵢ = (1 / k) Σ_{xⱼ ∈ Nₖ(xᵢ)} yⱼ

The behavior of k-NN is controlled by several design choices:

k (number of neighbors): small k can lead to overfitting and noisy decision boundaries; large k produces smoother boundaries but may underfit.

Distance metric: Euclidean or Manhattan distance can currently be used, with the optionality of adding other metrics in the future depending on need.

Weighting scheme: neighbors can contribute equally, or be weighted by distance so closer points have more influence on ŷᵢ.

Because k-NN makes predictions directly from the training data without learning an explicit parametric model, it is simple to implement, naturally handles multi-class problems, and can capture complex decision boundaries. However, it can be computationally expensive at prediction time and is sensitive to feature scaling, making preprocessing (e.g., normalization) important in practical applications.

---

## Data