# Spanning Tree
A [[spanning tree]] is a subset of a [[graph]] that is a tree which includes all the vertices of a graph.
![Spanning Tree|300](https://i0.wp.com/oliviagallucci.com/wp-content/uploads/2024/06/mst.webp?w=1076&ssl=1)

**Example**: Consider the graph below.
![image|200](https://notes-media.kthiha.com/Weighted-Graph/e0e9033b1dc8e1f2564fa4b9a8bb9d28.png)
The spanning trees of the graph are:
![image|200](https://notes-media.kthiha.com/Weighted-Graph/41ed08d9bad8d53ad6d91c3a9848ae9d.png)

---
### Properties of Spanning Trees
- [[Spanning tree|Spanning trees]] do not have any cycles and connected be disconnected.
- Every connected and undirected [[graph]] has at least one [[spanning tree]]. A disconnected [[graph]] can't have any [[Spanning Tree|spanning trees]]. Furthermore, a connected, undirected [[graph]] can have more than one [[spanning tree]].
- All possible [[Spanning Tree|spanning trees]] of a graph have the same number of vertices and edges.
- The [[spanning tree]] does not have any cycles.
- **Minimally Connected**:
  Removing one edge from a [[spanning tree]] will disconnect it.
  This means that a [[spanning tree]] is `minimally connected`.
- **Maximally Acyclic**:
  Adding one edge to a [[spanning tree]] will create a cycle.
  This means that a [[spanning tree]] is `maximally acyclic`.
- A [[spanning tree]] has $n-1$ edges where $n$ is the number of vertices.
- From a complete [[graph]], by removing max $e-n+1$ edges, we can construct a [[spanning tree]].
- Both [[Depth-First Search (DFS)|DFS]] and [[Breadth-First Search (BFS)|BFS]] creates [[Spanning Tree|spanning trees]].

---
## Min-Cost Spanning Tree (MST)
A min cost spanning tree is a [[spanning tree]] such that the sum of the [[Weighted Graph|weights]] is the smallest possible sum.

---
