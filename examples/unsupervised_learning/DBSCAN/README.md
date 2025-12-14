# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)


This directory contains example code and notes for the DBSCAN algorithm
in unsupervised learning.

Overview

DBSCAN is an unsupervised, density-based clustering algorithm used to group data points based on how closely they are packed together. Unlike K-Means, DBSCAN does not require specifying the number of clusters in advance and can identify arbitrarily shaped clusters while also detecting outliers (noise).

DBSCAN is especially useful when clusters vary in shape (i.e. they are not spherical) and when noise is present in the data.

Key Concepts

DBSCAN relies on two main parameters:
- eps (ε): The maximum distance between two points for them to be considered neighbors.
- min_samples (MinPts): The minimum number of points required to form a dense region.

Based on these parameters, points are classified as one of the follwoing:
- Core Point: A point with at least min_samples neighbors within eps.
- Border Point: A point that is within eps of a core point but has fewer than min_samples neighbors.
- Noise Point (Outlier): A point that is neither a core nor a border point.

Core and border points are then clustered into groups based on their density 

How DBSCAN Works
1. Select Parameters: Choose values for eps and min_samples.
2. Identify Core Points: For each point, count how many neighbors lie within distance eps. Distance is normally measured using the Euclidean norm
3. Form Clusters: Start a new cluster from a core point and then recursively add all density-reachable points to the cluster.
4. Expand Clusters: Continue expanding a given cluster until no more points can be added.
5. Mark Noise: Points that are not assigned to any cluster are labeled as noise.


Two important definitions:

Directly Density-Reachable
A point is directly density-reachable from a core point if it lies within eps.

Density-Reachable
A point is density-reachable if there exists a chain of core points connecting it.

These concepts allow DBSCAN to form clusters with complex shapes.
