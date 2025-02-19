# Graph Node Classification - Concepts

## Overview

This project implements Graph Neural Networks (GNN) for node classification tasks.
Node classification means predicting labels of nodes in a graph.

The implementation supports two main architectures:

- Graph Convolutional Networks (GCN)
- GraphSAGE

## Approach

First thing to do is to understand the dataset.
Since this is a node classification problem with 7 classes, we need to do multi level classification.
The dataset is already split into train, validation and test sets. So we can directly use it.

Implemented the basic pytorch train and evaluation loop.
Set the random seed for reproducibility.
Learned the basics of graph neural networks with pytorch geometric.
Defined the model architecture for both GCN and GraphSAGE inside models folder.

Usual multi level classification, uses cross entropy loss. But since this is a graph, used negative log likelihood (NLL) loss.
So added a log softmax layer at the end of the model.

Used the given parameters for learning rate, weight decay and hidden channels.
Added a config file to store all the parameters.

- For each epoch in Training Loop:

  1. Forward pass and loss calculation using NLL loss
  2. Backward pass and parameter updates
  3. Evaluation on train and validation sets
  4. Early stopping check based on configured metric (loss or accuracy)
  5. Save best model if monitored metric improves

Added early stopping to prevent overfitting. Its configured to be based on validation loss not decreasing even after 20 epochs (patience).

- Early Stopping

  1. Monitors validation metric (configurable: loss or accuracy)
  2. Saves best model state to configured checkpoint path
  3. Stops training if no improvement for specified patience period
  4. Maintains record of best metric


## Intial Results

![GCN Training](./docs/GCN.png)
![GraphSAGE Training](./docs/SAGE.png)

![Inference](./docs/inference.png)

For further improving the accuracy, use hyperparameter tuning tools such as optuna.

### Graph Neural Networks

GNNs are deep learning models designed to work with graph-structured data.
They learn node representations by aggregating information from neighboring nodes and their features.

Graphs can be represented as an Adj matrix.
But we cannot apply traditional neural networks because:

- Graphs may be of different sizes
- Adj matrix is not invariant to ordering of nodes.

X = Node features x number of nodes.
A = Adjacency matrix.

Computation Graph: Neighbour of a node defines the computation graph
Every node has its own computation graph.
To predict the label of a node, we need to order invariant aggregate (sum, average, max, min) the features of the neighbours.
We don't need to unroll the graph too much, because we are only interested in the immediate neighbours.
The weights are shared across all nodes. So convolution operation can work well, as the kernel weights can be applied to all nodes.

GraphSage: Modified to include a general aggregation function and instead of summing the previous feature representation, we concatenate it.
Aggregation can be a pooling strategy like element wise min, max etc.

Many ways to create node features such as Random Walk, Deep Walk, Node2Vec.

## Data Exploration

### CORA from planetoid dataset.

It contains papers and their citations.
There are 7 classes : Reinforcement Learning, Case Based, Theory, Genetic Algorithms, Neural Networks, Probabilistic Methods, Rule Learning.
There are 2708 papers (nodes) and 5429 citations (edges).

Train Mask [2708 * 1] contains boolean values indicating whether a node is in the training set or not.
Same for validation and test masks.
There are 140 training nodes, 500 validation nodes and 1000 test nodes.

X is the node features matrix. It is a 2708 _ 1433 matrix. 1433 is the number of features for each node. It contains binary values for 1433 words
Y is the node labels matrix. It is a 2708 _ 7 matrix. 1 to 7 values

90+ Accuracy (https://paperswithcode.com/sota/node-classification-on-cora) is the state of the art model.

## Pytorch Basics

### Dataset Preparation

1. Load the dataset
2. Split the dataset into train, validation and test sets
3. Create a DataLoader for each set
4. Move the data to the device (CPU or GPU)

### Model Creation

1. Define the model architecture
2. Move the model to the device (CPU or GPU)
3. Define the loss function
4. Define the optimizer

### Basic Training Loop

1. Set the model to train()
2. Zero the gradients (optimizer.zero_grad())
3. Forward pass (out = model(data))
4. Calculate loss (loss = LossFunction(out, target))
5. Backward pass (loss.backward())
6. Update parameters (optimizer.step())

### Basic Evaluation Loop

1. Set the model to eval()
2. Ensure no gradients are calculated (@torch.no_grad())
3. Forward pass (out = model(data))
4. Calculate accuracy and loss

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
