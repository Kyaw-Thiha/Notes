# Spanning Tree
A [[Spanning Tree]] is a subset of a [[Graph]] that is a tree which includes all the vertices of a graph.
![Spanning Tree|300](https://i0.wp.com/oliviagallucci.com/wp-content/uploads/2024/06/mst.webp?w=1076&ssl=1)

**Example**: Consider the graph below.
![image|200](https://notes-media.kthiha.com/Weighted-Graph/e0e9033b1dc8e1f2564fa4b9a8bb9d28.png)
The spanning trees of the graph are:
![image|200](https://notes-media.kthiha.com/Weighted-Graph/41ed08d9bad8d53ad6d91c3a9848ae9d.png)

---
### Properties of Spanning Trees
- [[Spanning Tree|Spanning trees]] do not have any cycles and connected be disconnected.
- Every connected and undirected [[Graph]] has at least one [[Spanning Tree]]. A disconnected [[Graph]] can't have any [[Spanning Tree|spanning trees]]. Furthermore, a connected, undirected [[Graph]] can have more than one [[Spanning Tree]].
- All possible [[Spanning Tree|spanning trees]] of a graph have the same number of vertices and edges.
- The [[Spanning Tree]] does not have any cycles.
- **Minimally Connected**:
  Removing one edge from a [[Spanning Tree]] will disconnect it.
  This means that a [[Spanning Tree]] is `minimally connected`.
- **Maximally Acyclic**:
  Adding one edge to a [[Spanning Tree]] will create a cycle.
  This means that a [[Spanning Tree]] is `maximally acyclic`.
- A [[Spanning Tree]] has $n-1$ edges where $n$ is the number of vertices.
- From a complete [[Graph]], by removing max $e-n+1$ edges, we can construct a [[Spanning Tree]].
- Both [[Depth-First Search (DFS)|DFS]] and [[Breadth-First Search (BFS)|BFS]] creates [[Spanning Tree|spanning trees]].

---
## Min-Cost Spanning Tree (MST)
A min cost spanning tree is a [[Spanning Tree]] such that the sum of the [[Weighted Graph|weights]] is the smallest possible sum.

---
