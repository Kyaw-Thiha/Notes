# Graph Attention Network
In [[Graph Attention Network(GAT)|GAT]], instead of using node degree, learn an attention score between two nodes. Learn the contribution weight of neighbour nodes.

![300](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1-Bvdnu6-Z2z9brLAOf2anpaATdpa90vWbNp7FdXaIA&s=10)

In order to do this,
- Use a shared neural network to compute an [[Self-Attention|attention score]] between two nodes.

$$
e_{ij} = \text{NN}(h_{i}, h_{j})
$$

- Normalize the attention score.

$$
\alpha_{ij} = \text{softmax}_{j}(e_{ij})
= \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_{i}} \exp(e_{ik})}
$$

- Update the node embeddings based on the attention score.

$$
h_{i} = \sigma \left( \sum_{j \in \mathcal{N}_{i}} \alpha_{ij} \mathbf{W} h_{j} \right)
$$

---
## See Also
- [Good But Long Blog](https://distill.pub/2021/gnn-intro/)
- [[Graph Neural Network(GNN)]]
- [[Graph Convolutional Network(GCN)]]
- [[Graph Attention Network(GAT)]]
- [[Self-Attention]]
- [[Attention Mechanism]]
- [[Transformer]]
