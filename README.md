# INDE-577-F25
Machine Learning Package for CMOR 438/ INDE 577 Data Science Machine Learning Fall 2025

In this repository, we implement a machine learning package that includes KNN, linear regression, neural network, perceptron, coomunity detection, DBSCAN, k means clustering, decision and regression trees, and ensemble methods for decision trees.

# Authors
Jonathan Woo
jjw11@rice.edu

Vikram Shah
vs56@rice.edu/ vikramshah1108@gmail.com

## Overview
This repository contains a machine learning toolkit called `rice_ml` which implements supervised and unsupervised learning algorithms.

## Features
- **Supervised learning**: k-nearest neighbors, linear and logistic regression, perceptron, feedforward neural network, decision trees, and regression trees.
- **Unsupervised learning**: k-means clustering, DBSCAN, principal component analysis (PCA), and community detection.
- **Data utilities**: preprocessing helpers and postprocessing utilities for common workflows.
- **Examples and tests**: focused examples for each algorithm and pytest coverage to validate behaviors.

## Installation
1. Ensure you have Python **3.9+**.
2. Clone the repository 
3. Install the package and core dependencies:
   `pip install -e .`


## Repository Structure
```
INDE-577-F25/
├── data/                      # Sample datasets 
├── examples/                  # Usage walkthroughs for supervised and unsupervised algorithms
├── src/
│   ├── rice_ml/               # Source package
│   │   ├── supervised_learning/  # KNN, regressions, perceptron, neural network, trees, metrics, preprocessing
│   │   └── unsupervised_learning/ # K-means, DBSCAN, PCA, community detection
│   └── README.md              # Notes about the source layout
├── tests/                     # Pytest suites for all algorithms and utilities
├── pyproject.toml             # Project metadata, dependencies, and tooling config
└── README.md                  # (This file) project overview and usage
```

## Usage
After installation, import algorithms directly from the package. Here is an example:
```python
from rice_ml.supervised_learning.knn import KNNClassifier
from rice_ml.unsupervised_learning.k_means_clustering import KMeansClustering

model = KNNClassifier(k=5)
```
Refer to the `examples/` folder for runnable walkthroughs tailored to each algorithm.

## Testing
Run the full test suite with pytest. Run the following command in the root directory of this repository:
```bash
pytest tests/
```

