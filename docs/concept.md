# Graph Node Classification - Concepts

## Overview

This project implements Graph Neural Networks (GNN) for node classification tasks.
Node classification means predicting labels of nodes in a graph.

The implementation supports two main architectures:

- Graph Convolutional Networks (GCN)
- GraphSAGE

## Core Concepts

### Graph Neural Networks

GNNs are deep learning models designed to work with graph-structured data. They learn node representations by aggregating information from neighboring nodes and their features.

### Model Architectures

#### 1. Graph Convolutional Network (GCN)

- Performs neighborhood aggregation using spectral graph convolutions
- Architecture:
  ```
  Input → GCNConv → ReLU → Dropout → GCNConv → ReLU → Dropout → Linear → LogSoftmax → Output
  ```
- Each GCNConv layer updates node features based on first-order neighbors

#### 2. GraphSAGE

- Inductive framework that generates embeddings by sampling and aggregating features from nodes' local neighborhoods
- Architecture:
  ```
  Input → SAGEConv → ReLU → Dropout → SAGEConv → ReLU → Dropout → Linear → LogSoftmax → Output
  ```
- Supports different aggregation functions:
  - Mean: Average of neighbor features (default)
  - Max: Element-wise max pooling
  - Sum: Sum of neighbor features
