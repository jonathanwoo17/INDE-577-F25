# Decision Trees (Entropy-Based)

## Overview

Decision Trees are a supervised machine learning algorithm used for both **classification** tasks. 
An entropy-based decision tree builds a model in the form of a tree structure
by recursively splitting the data to maximize **information gain**, a quantity derived from
**entropy**.



## Core Idea

The key idea behind entropy-based decision trees is to choose splits that result in the **purest**
possible child nodes.

- **Entropy** measures the level of uncertainty or impurity in a dataset
- **Information Gain** measures how much entropy is reduced after a split
- The best split is the one that **maximizes information gain**



## Entropy

Entropy quantifies the randomness or disorder in a dataset.

For a dataset \(S\) with class probabilities \(p_1, p_2, \ldots, p_k\):

\[
H(S) = - \sum_{i=1}^{k} p_i \log_2 p_i
\]

Properties of entropy:
- \(H(S) = 0\) when the dataset is perfectly pure (all samples in one class)
- Entropy is maximized when classes are evenly distributed



## Information Gain

Information Gain measures the reduction in entropy after splitting the dataset on a feature.

\[
IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
\]

Where:
- \(S\) is the original dataset
- \(A\) is the feature used for splitting
- \(S_v\) is the subset of samples where feature \(A\) takes value \(v\)

The feature with the highest information gain is chosen for the split.

---

## How Entropy-Based Decision Trees Work

1. **Start at the Root**
   - Compute the entropy of the target labels.

2. **Evaluate All Possible Splits**
   - For each feature, compute the information gain.

3. **Choose the Best Split**
   - Select the feature with the highest information gain.

4. **Split the Dataset**
   - Partition the data into subsets based on the chosen feature.

5. **Repeat Recursively**
   - Apply the same process to each child node.

6. **Stopping Criteria**
   - All samples belong to the same class
   - No features remain
   - Maximum tree depth is reached
   - Minimum number of samples per node is reached

The tree then uses the "decisions" it has found to classify new data.
