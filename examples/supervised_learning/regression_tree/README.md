# Regression Trees

## Overview

Regression Trees are a type of **supervised learning** algorithm used for predicting **continuous
numerical values**. They are a variant of decision trees where the target variable is continuous
rather than categorical.


---

## Core Idea

The main idea behind regression trees is to split the data in a way that minimizes **prediction
error** within each region.

- The dataset is recursively divided into smaller subsets
- Each split is chosen to minimize the **mean squared error (MSE)**
- Leaf nodes store a numerical prediction value

---

## Splitting Criterion

Regression trees typically use **Mean Squared Error (MSE)** or **variance reduction** to select the
best split.

### Mean Squared Error (MSE)

For a dataset \(S\) with target values \(y_1, y_2, \ldots, y_n\):

\[
\text{MSE}(S) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2
\]

Where:
- \(\bar{y}\) is the mean of the target values in \(S\)

### Variance Reduction

A split is chosen to minimize the weighted MSE of the child nodes:

\[
\Delta = \text{MSE}(S) -
\left(
\frac{|S_L|}{|S|}\text{MSE}(S_L)
+
\frac{|S_R|}{|S|}\text{MSE}(S_R)
\right)
\]

The split with the largest reduction in error is selected.

---

## How Regression Trees Work

1. **Start at the Root**
   - Compute the error (variance or MSE) of the target values.

2. **Evaluate All Possible Splits**
   - For each feature and split point, compute the reduction in error.

3. **Choose the Best Split**
   - Select the split that minimizes the prediction error.

4. **Split the Dataset**
   - Partition the data into left and right child nodes.

5. **Repeat Recursively**
   - Apply the same process to each child node.

6. **Stopping Criteria**
   - Maximum tree depth reached
   - Minimum number of samples per leaf
   - No significant reduction in error

---

## Prediction

For a new input, a regression tree 
- Traverses the tree based on feature values
- Reaches a leaf node
- Outputs the mean of training samples in that leaf

