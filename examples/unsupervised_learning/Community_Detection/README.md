# Community Detection

This directory contains example code and notes for the Community Detection algorithm 
in unsupervised learning.

Overview

This Community Detection implementation uses Label Propagation. Community Detection is an unsupervised algorithm used for finding groups of related nodes in graphs and networks. It identifies communities by propagating labels through the network based on node connectivity.

Community Detection is popular because it is simple, fast, and scalable, making it suitable for large networks such as social graphs, citation networks, and communication networks.

Core Idea

Each node in the graph holds a label representing its community.
Through iterative updates, nodes adopt the label that is most common among their neighbors.
Over time, densely connected nodes converge to the same label, forming communities.

How Label Propagation Works
1. Initialization: Assign a unique label to every node in the graph.
2. Label Update: For each node, look at the labels of its neighbors and update the node’s label to the most frequent label among its neighbors. Break ties randomly
3. Iteration: Repeat the update process for all nodes. Updates are often done in random order to avoid bias.
4. Convergence: The algorithm stops when the change in labels is below a user set threshold or when a maximum number of iterations is reached.
5. Each unique label at convergence represents a community.

Advantages of Community Detection are:
- No need to predefine the number of communities
- Fast for large graphs
- Simple to implement for basic versions of Community Detection