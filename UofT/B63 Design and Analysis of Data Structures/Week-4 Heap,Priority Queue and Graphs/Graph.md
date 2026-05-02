# Graph
A [[graph]] $G = (V, E)$ consists of a set of $\text{vertices(nodes)}$, denoted by $V$ and a set of $\text{edges}$, denoted by $E$.
- $n = |V|$ is the number of nodes.
- $m = |E|$ is the number of edges.

In an `undirected graph`, each edge is a set of $2$ vertices, $\{ u, v \}$
- This makes $(u, v)$ and $(v, u)$ the same.
- Furthermore, self-loops are not allowed.

In a `directed graph`, each edge is an ordered pair of nodes.
- Therefore, $(u, v)$ is different from $(v, u)$.
- Furthermore, self-loops are allowed. This means $(u,u)$ is allowed.

---
## Graph Representation
Consider the graph below.
![image|200](https://notes-media.kthiha.com/Graph/9b29adbdd834d8105221910ac224ecf2.png)

### Adjacency Matrix
An [[#adjacency matrix]] is a 2D array.
![image|300](https://notes-media.kthiha.com/Graph/b78c8bf8bac278a105289c18630f07df.png)

- Space: $\Theta(n^{2})$
- Who are adjacent to $v$: $\Theta(n)$
- Are $v$ and $w$ adjacent: $\Theta(1)$

> Convenient for some other operations and queries.

---
### Adjacency List
With [[#adjacency list]], we store the vertices in a 1D array or dictionary. At entry $A[i]$, we store the neighbours of $V_{i}$.
![image|300](https://notes-media.kthiha.com/Graph/f6ef5e7e4865cc4b88e494b91f7028e8.png)
If the graph is directed, we store only the out-neighbours.
- Space: $\Theta(m + n)$
- Who are adjacent to $v$: $\Theta(deg(v)) \text{ time}$ (length of adj list)
- Are $v$ and $w$ adjacent: $\Theta(deg(v))$ time if a list

> Optimal for graph searches.

---
## Terminologies
### Path
A sequence of edges which connect a sequence of distinct vertices. (You can't go through a vertex twice)

![image|300](https://notes-media.kthiha.com/Graph/554202bd365ddcbf39b8e6a44375cf3e.png)

- **Transversal**: Visit each vertex of a [[graph]].
- **Reachability**: $v$ is reachable from $u$ $iff$ there is a path from $u$ to $v$

---
### Simple Cycle
A [[#simple cycle]] is a non-empty sequence of vertices in which
- Consecutive vertices are adjacent.
- First vertex = Last vertex
- Vertices are distinct, except for the first and last.
- Edges used are distinct.

Note that $<v>$ is not a cycle.
![image|300](https://notes-media.kthiha.com/Graph/044da2fc32d6579ddb889711f0e4aec9.png)

---
### Tree
A tree is a [[graph]] that is connected but has no cycles.
![image|300](https://notes-media.kthiha.com/Graph/1a3badad46c6bc92df4e232cd47792ea.png)
A forest is a collection of trees.
Note: `Acyclic` means that there are no cycles.

Trees have the following properties:
- Between any $2$ vertices, there is a unique path.
- A tree is connected by default, but if an edge is removed, it becomes disconnected.
- $\text{no. of edges} = \text{no. of vertices} - 1$ ($m=n-1$)
- Acyclic by default.
  But if a new edge is added, then it will have a cycle.

---
### Weighted Graph
Each edge in the [[graph]] is assigned a real number called its weight.

---
### Connectivity
- `Connected`: For undirected graphs, every $2$ vertices have a path between them.
- `Strongly Connected`: For directed graphs, for any $2$ vertices $u,v$, there is a directed path from $u$ to $v$.

![image|300](https://notes-media.kthiha.com/Graph/70e6c710965bec238c3b5faf08c40f6b.png)

---
### Operations
- `Add/Remove` a vertex/edge.
- `Edge Query`: Given $2$ vertices $u$ and $v$, find out if the $edge(u,v)$ (if the graph is directed) or the edge $\{ u,v \}$ is in $E$
- `Neighbourhood`: Given a vertex $v$ in an undirected graph, get the set of vertices $\{ v \mid \{ u,v \} \in E \}$
- `In-neighbourhood`: Given a vertex $v$ in a directed graph, get the set of vertices $\{ v \mid (v,u) \in E \}$
  This gets the set of vertices whose edges lead to $u$
- `Out-neighbourhood`: Given a vertex $u$ in a directed graph, get the set of vertices $\{ v \mid (u,v) \in E \}$
  This gets the set of vertices that can be reached by the edges that lead away from $u$.
- `Degree`: Computes the size of the neighbourhood.
- `In-Degree`: Computes the size of the in-neighbourhood.
- `Out-Degree`: Computes the size of the out-neighbourhood.

---
## See Also
- [[Breadth-First Search (BFS)]]
- [[Priority Queue]]
