# Depth-First Search
You start at a node $v$ and go all the way down a path until you hit a dead-end. Then, you backtrack and go down other paths.

---
## Algorithm
- All vertices and edges start out unmarked.
- Start at vertex $v$ and go as far as possible away from $v$ visiting vertices.
- If the correct vertex has not been visited, mark it as visited and the edge that is transversed as a [[Depth-First Search (DFS)|DFS]] edge.
- If the current vertex has been visited, mark the transversed edge as a back-up edge.
  Then, move back up to the previous vertex.
- When the current vertex has only visited neighbours left, mark it as finished.
- Back-track to the first vertex that is not finished.
- Continue

---
## Properties
- Like a [[Breadth-First Search (BFS)|BFS]], a [[Depth-First Search (DFS)|DFS]] also creates a non-unique spanning tree.
- A [[Depth-First Search (DFS)|DFS]] also gives connected component information.
- Unlike a [[Breadth-First Search (BFS)|BFS]], a [[Depth-First Search (DFS)|DFS]] does not find the shortest path between $2$ nodes. A [[Depth-First Search (DFS)|DFS]] is faster than a [[Breadth-First Search (BFS)|BFS]].

---
## Implementing a DFS
We can use a stack(LIFO) to store the edges with the operations:
- `push((u,v))`
- `pop()`
- `is_empty()`

Furthermore, we need to store these data in each node in order to determine whether an edge is a back-edge or a DFS edge:
- $d[v]$: Discovery Time
- $f[v]$: Finish Time

---
## Complexity
Since [[Depth-First Search (DFS)|DFS]] visits the neighbourhoods of a node exactly once, the adjacency list of each vertex is visited at most once.
Therefore, the total running time is $O(m + n)$.

---
## DFS Edges
The [[Depth-First Search (DFS)|DFS]] edges form a tree called the [[Depth-First Search (DFS)|DFS tree]].
However, the [[Depth-First Search (DFS)|DFS tree]] is NOT unique for a given graph $G$ starting at $S$.

- We can specify edges $(u,v)$ in a [[Depth-First Search (DFS)|DFS tree]] according to how they are transversed during the search.
- If $v$ is visited for the first time, then $(u,v)$ is a `tree-edge` in a [[Depth-First Search (DFS)|DFS tree]].
- If $v$ has already been visited, then $(u,v)$ is a 
	- `back-edge`: An edge from a vertex $u$ to an ancestor $v$ in the [[Depth-First Search (DFS)|DFS tree]]
	- `forward-edge`: An edge from a vertex $u$ to a descendent $v$ in the [[Depth-First Search (DFS)|DFS tree]].
	  **Note**: This only applies to [[Graph|directed graphs]].
	- `cross-edge`: All the other edges that are not part of the [[Depth-First Search (DFS)|DFS tree]]. $v$ is neither an ancestor nor a descendent of $u$ in the [[Depth-First Search (DFS)|DFS tree]].
	  **Note**: This only applies to [[Graph|directed graphs]].
	![image|300](https://notes-media.kthiha.com/Depth-First-Search-(DFS)/974f2be07524df7f7c99b6c72b95c579.png)
- We can use $d[v]$ and $f[v]$ to distinguish between the edges.
- There is a cycle in [[graph]] $G$ $\iff$ there are any `back-edges` when [[Depth-First Search (DFS)|DFS]] is run.
- We can detect a `back-edge` in a [[Depth-First Search (DFS)|DFS]] if the vertex we are visiting has been visited but not finished.

---
## Example
Consider the graph below.
![image|300](https://notes-media.kthiha.com/Depth-First-Search-(DFS)/3b90e0eed4a17fd4c183e3f94699d818.png)


---
## See Also
- [[Graph]]
- [[Breadth-First Search (BFS)]]
- [[Kosaraju's SCC Algorithm]]