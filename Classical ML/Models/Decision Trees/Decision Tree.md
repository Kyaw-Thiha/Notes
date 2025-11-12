# Decision Tree
#ml/models/classic/decision-tree
`Decision Trees` partition the dataset using threshold from the tree structure.

![Decision Tree](https://blog.mindmanager.com/wp-content/uploads/2022/03/Decision-Tree-Diagram-Example-MindManager-Blog.png)

---
## Split Function Threshold

At each node, we use a `Univariate Variate Split Functions` over single feature at each node.
The goal is to learn the optimal threshold $\tau_{j}$
$$
t_{j}(x) = e_{l}^T x > \tau_{j}
$$
where
- $e_{l} = [0, 0, \dots, 1, 0, \dots, 0]$ is the `one-hot vector` where only $l^{th}$ feature is 1 such that $e_{l}^T x = x_{l}$
- $\tau_{j}$ is the `optimal threshold`

## Why Univariate Threshold?

Choosing a threshold midway between two adjacent points makes learning more robust to noise.
At each feature $j$ (also node $j$), there are $(N_{j}-1)$ possible thresholds.
This means that for $K$ features, we will have $(N_{j} - 1)^K$ possible threshold.

## Choosing optimal threshold
Compute [[Information Gain]] over each of the features $j$.
Choose the feature threshold with highest `Information Gain`.
This is done through the [[Greedy Algorithm]].

---
## Decision Tree vs [[K-NN]]
### `Decision Boundary`
`K-NN` learns complicated decision boundary.
`Decision Tree` learns rectangular regions of decision boundaries. It is always split parallel to the feature dimension.

### `Information Storage`
`K-NN` requires keeping the entire dataset, hence needs to search through huge amount of data.
`Decision Tree` only keeps split functions, and the probability distribution over each class labels for each leafs.

---