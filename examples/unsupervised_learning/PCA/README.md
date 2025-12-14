# Principal Component Analysis (PCA)

## Overview

**Principal Component Analysis (PCA)** is an unsupervised dimensionality reduction technique used to simplify high-dimensional data while preserving as much variance as possible. PCA transforms the original features into a smaller set of **uncorrelated variables** called **principal components**.

PCA is widely used for data visualization, noise reduction, and improving the performance of machine learning models.


## Core Idea

PCA finds new axes (principal components) that are ordered by the amount of variance they explain  
- The first principal component explains the most variance, the second explains the next most, and so on.


## How PCA Works

1. **Standardize the Data**: Features are scaled to have zero mean and unit variance.
2. **Compute the Covariance Matrix**: Measures how features vary with respect to each other.
3. **Eigen Decomposition**: Compute eigenvectors (directions) and eigenvalues (variance) of the covariance matrix.
4. **Select Principal Components**: Choose the top `k` eigenvectors with the largest eigenvalues.
5. **Project the Data**: Transform the original data into the new lower-dimensional space.

## PCA and Singular Value Decomposition (SVD)

**Singular Value Decomposition (SVD)** is a matrix factorization technique that decomposes a data matrix \(X\) into three components:

\[
X = U \Sigma V^T
\]

Where:
- **U** contains the left singular vectors  
- **Σ (Sigma)** is a diagonal matrix of singular values  
- **Vᵀ** contains the right singular vectors  



### Relationship Between PCA and SVD

PCA is computed using **SVD**.
When PCA is applied to a **mean-centered data matrix**:
- The **singular values** are related to the **explained variance**
- The **principal component scores** are obtained from \(U \Sigma\)

In practice:
- Eigen-decomposition of the covariance matrix and SVD of the data matrix yield equivalent PCA results
- SVD is preferred due to **numerical stability** and **computational efficiency**, especially for large datasets


### Explained Variance via SVD

The variance explained by each principal component is proportional to the **square of the singular values**:

\[
\text{Explained Variance}_i \propto \sigma_i^2
\]

## PCA Advantages

- Reduces dimensionality and storage requirements  
- Improves visualization (2D or 3D plots)  
- Can improve model training speed and performance  




