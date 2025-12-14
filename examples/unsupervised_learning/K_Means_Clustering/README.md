# K Means Clustering

This directory contains example code and notes for the K Means Clustering algorithm
in unsupervised learning.

Overview

K-Means clustering is an unsupervised machine learning algorithm used to group data points into K distinct clusters based on similarity. Each cluster is represented by a centroid, which is the mean position of all points in that cluster.

The goal of K-Means is to minimize the within-cluster variance, meaning that data points in the same cluster should be as similar as possible.

How K-Means Works

The algorithm follows an iterative process:

1. Choose K

Decide how many clusters (K) you want to find in the data.

2. Initialize Centroids

Randomly place K centroids in the feature space (or use a smarter method like k-means++).

3. Assign Points to Clusters

Each data point is assigned to the nearest centroid, using Euclidean or Manhatten distance.

4. Update Centroids

For each cluster, recompute the centroid as the mean of all assigned points.

5. Repeat

Steps 3 and 4 are repeated until:

Centroids no longer move significantly, or

A maximum number of iterations is reached.
