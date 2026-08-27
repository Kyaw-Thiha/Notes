# Graph Convolutional Network

A layer of [[Graph Neural Network(GNN)|GNN]] is a non-linear function over node features and adjacency matrix.

![image|400](https://notes-media.kthiha.com/Graph-Neural-Network(GNN)/80660da7bf9d12cd55f85e651eb6ac8c.png)

It can be defined as
$$
\mathbf{H} = \text{ReLU}(\mathbf{AXW} + b)
$$
But this has $2$ certain limitations.

---
## Self-Loops
**Limitation**: Multiplication with $\mathbf{A}$ means that, for every node, we sum up all the feature vectors of all neighboring nodes but not the node itself.

**Fix**: Add self-loops. Add the identity matrix to $A$.
$$
\mathbf{A} = \mathbf{A} + \mathbf{I}
$$

---
## Symmetric Normalization
**Limitation**: $\mathbf{A}$ is not normalized and therefore the multiplication with $\mathbf{A}$ will completely change the scale of the feature vectors.

**Fix**: Symmetrically normalize $\mathbf{A}$ using diagonal degree matrix $\mathbf{D}$ such that all rows sum to one.
$$
\mathbf{A}
= \mathbf{D}^{-1/2} \ \mathbf{A} \ \mathbf{D}^{-1/2}
$$

---
## GCN 
With the above-mentioned fixes, we can define the [[Graph Convolutional Network(GCN)|GCN layer]] as
$$
\mathbf{H} = \text{ReLU}(\mathbf{D}^{-1/2} \ \mathbf{A} \mathbf{D}^{-1/2} \ \mathbf{X} \ \mathbf{W} + b)
$$
where
- $\mathbf{H}$: output node embeddings.
- $\mathbf{A}$: adjacency matrix of graph.
- $\mathbf{D}$: degree matrix correponding to $\mathbf{A}$.
  A diagonal matrix where $\mathbf{D}_{ii}$​ is the sum of row $i$ of $\mathbf{A}$.
- $\mathbf{X}$: input node feature matrix from prev layer.
- $\mathbf{W}$: learnable weight matrix for this layer.

---
# Going Deeper
A [[Graph Convolutional Network(GCN)|GCN layer]] updates the node embeddings based on the features of the immediate neighbors(multiplication with $\mathbf{A}$).

![image|500](https://notes-media.kthiha.com/Graph-Convolutional-Network(GCN)/ecf89be8a21b94828c1fc0a0c4e0166b.png)

By stacking [[Graph Convolutional Network(GCN)|GCN layers]], we can influence embeddings from further neighbourhood. Note that this is analogous to increasing the receptive field in [[Convolutional Neural Network (CNN)|CNNs]].

---
## See Also
- [Good But Long Blog](https://distill.pub/2021/gnn-intro/)
- [[Graph Neural Network(GNN)]]
- [[Graph Convolutional Network(GCN)]]
- [[Graph Attention Network(GAT)]]
