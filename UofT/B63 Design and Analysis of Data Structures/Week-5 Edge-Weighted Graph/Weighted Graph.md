# Edge-Weighted Graph
A [[graph]] where each edge has a weight associated to it.

We denote the weight of an edge by $w(e)$.
Usually $w(e) \geq 0$, but it can also be less than $0$.

![Weighted Graph|300](https://i.ytimg.com/vi/cMijJ2C1TiI/maxresdefault.jpg)

An [[Weighted Graph|edge-weighted graph]] consists of 
- A set of vertices $V$
- A set of edges $E$
- Weights: A map from edges to numbers $w: E \to \mathbb{R}$
  **Note**: 
	- For [[Graph|undirected graphs]], $\{ v,u \} = \{ u,v \}$ so they have the same weight.
	- For [[Graph|directed graphs]], $(u,v)$ and $(v,u)$ may have different weights.
- Notations: In addition to $w(e)$, we may also use $w(u,v)$ and $\text{weight}(u,v)$ to denote the weight of an edge.

---
## Storing a Weighted Graph
We may use adjacency list or adjacency matrix to store a weighted graph.

**Example**: Consider the [[weighted graph]] below.
![image|200](https://notes-media.kthiha.com/Weighted-Graph/05e32ba3efe1b710eae7b584865baec0.png)
![image|200](https://notes-media.kthiha.com/Weighted-Graph/ef8e3c419628d97effaa2f6e661a4d7f.png)
![image|200](https://notes-media.kthiha.com/Weighted-Graph/d5febac0abf5c67c175e99b3d147dc75.png)

---
## See Also
- [[Spanning Tree]]
- [[Minimum Spanning Tree (MST)]]
- [[Kruskal's Algorithm]]
- [[Prim's Algorithm]]
- [[Dijkstra's Algorithm]]
