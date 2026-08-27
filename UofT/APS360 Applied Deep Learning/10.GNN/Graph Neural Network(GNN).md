# Graph Neural Network
A [[Graph Neural Network(GNN)|GNN]] is a [[Neural Network|network]] that optimize transformations on all attributes of the graph(nodes,edges,global context) and preserve graph symmetries(permutation invariances).

![image|500](https://notes-media.kthiha.com/Graph-Neural-Network(GNN)/8a877758852a94df27524f80d87db724.png)

---
## Message Passing
[[Graph Neural Network(GNN)|GNNs]] learn embeddings over [[Graph|graphs]] through [[#Message Passing|message passing]].

![500](https://syedarizvi.com/assets/img/blog-20240609-gnn-basics-3-message-passing.png)

For each node in graph,
- **Aggregate** embeddings of its neighbour nodes.
- **Combine** the aggregated embedding with node embedding.
- **Update** the node embedding.

![300](https://www.researchgate.net/publication/372486484/figure/fig2/AS:11431281201950942@1698671714642/GNN-basic-structure-with-message-transformation-aggregation-and-update-52.png)

This can be formulated as
$$
h_{v}^{(k)}
= \text{COMBINE}(h_{v}^{(k-1)}, \ \text{AGGREGATE}(\{ h_{u}^{(k-1)}: u \in \mathcal{N}(v) \}))
$$

![500](https://dmol.pub/_images/gcn.gif)

[[Graph Neural Network(GNN)|Aggregate function]] must be order-invariant function such as sum, mean, max, [[Attention Mechanism|attention]], etc.

---
## Read-Out(Graph-Pooling) Function

![image|400](https://notes-media.kthiha.com/Graph-Neural-Network(GNN)/8a877758852a94df27524f80d87db724.png)

- Aggregates all node embeddings into a graph embedding.
- Must be order-invariant function such as sum, mean, max, attention, etc.

![300](https://media.springernature.com/lw1200/springer-static/image/art%3A10.1007%2Fs10462-024-10918-9/MediaObjects/10462_2024_10918_Fig2_HTML.png)

---
## GNN Tasks
[[Graph Neural Network(GNN)|GNN]] can do different prediction tasks.

![image|500](https://notes-media.kthiha.com/Graph-Neural-Network(GNN)/11dccf9a81c7fedcc08dcf3186350d9b.png)

---
## Mathematical Formulation
A layer of [[Graph Neural Network(GNN)|GNN]] is a non-linear function over node features and adjacency matrix.

![image|400](https://notes-media.kthiha.com/Graph-Neural-Network(GNN)/80660da7bf9d12cd55f85e651eb6ac8c.png)

It can be defined as
$$
\mathbf{H} = \text{ReLU}(\mathbf{AXW} + b)
$$
After certain mathematical fixes through [[Graph Convolutional Network(GCN)|GCN]], we get
$$
\mathbf{H} = \text{ReLU}(\mathbf{D}^{-1/2} \ \mathbf{A} \mathbf{D}^{-1/2} \ \mathbf{X} \ \mathbf{W} + b)
$$
We could then stack these layers to increase neighbours influencing the embedding.

![image|500](https://notes-media.kthiha.com/Graph-Convolutional-Network(GCN)/ecf89be8a21b94828c1fc0a0c4e0166b.png)

---
## Dense/Sparse Implementation
### Dense Implementation
This uses the [[Graph|adjacency matrix]].

![image|500](https://notes-media.kthiha.com/Graph-Neural-Network(GNN)/6eca547b084a03647740b79e69757765.png)

However, 
- Most [[Graph|graphs]] are sparse.
- Dense implementation uses $N^{2}$ space for adjacency matrix.
- We can represent the graph above with only $4$ edges where dense implementation represents it with $25$.

---
### Sparse Implementation

![image|500](https://notes-media.kthiha.com/Graph-Neural-Network(GNN)/ae38848cc0988f0dafa027f422e83e57.png)

---
### Batching

![image|500](https://notes-media.kthiha.com/Graph-Neural-Network(GNN)/8aaeece3ef56a24cef9a3666d64ab881.png)

![image|500](https://notes-media.kthiha.com/Graph-Neural-Network(GNN)/2a442598c8a37862af566aa8c1824f32.png)

---
## See Also
- [Good But Long Blog](https://distill.pub/2021/gnn-intro/)
- [[Graph Neural Network(GNN)]]
- [[Graph Convolutional Network(GCN)]]
- [[Graph Attention Network(GAT)]]