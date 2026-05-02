# Breath-First Search
You start at vertex $v$, go to all its unvisited neighbours.
Then, repeat for all of $v$'s neighbours, and so on.
[[Breadth-First Search (BFS)|BFS]] will find the shortest distance between $2$ vertices.

---
## Algorithm
- Start at $v$.
  Visit $v$ and mark it as visited.
- Visit every unmarked neighbour of $v$ and mark every neighbour as visited.
- Mark $v$ as finished.
- Recurse on each vertex marked as visited in the order they were visited.

---
## Example
![image|300](https://notes-media.kthiha.com/Breadth-First-Search-(BFS)/6ae32d572701cd7c3514aca193075365.png)

![image|300](https://notes-media.kthiha.com/Breadth-First-Search-(BFS)/618548d33a3aabf36c6c5ff0f7056f2d.png)
![image|300](https://notes-media.kthiha.com/Breadth-First-Search-(BFS)/b0e06940864787d3857d52e48b4a7803.png)
![image|300](https://notes-media.kthiha.com/Breadth-First-Search-(BFS)/aef494ae5ddca2a9fea720faac127aa3.png)
![image|300](https://notes-media.kthiha.com/Breadth-First-Search-(BFS)/5ea3f5bd36afc6cd528d2aa9fec7fe93.png)
![image|300](https://notes-media.kthiha.com/Breadth-First-Search-(BFS)/bb7ae28ff5a963f2c86cab0fe528d5f1.png)
![image|300](https://notes-media.kthiha.com/Breadth-First-Search-(BFS)/27a67f37d9ce31a1d75e3d7d39fa19b8.png)
![image|300](https://notes-media.kthiha.com/Breadth-First-Search-(BFS)/388b7b36f687bddd55863ee0add8af18.png)
![image|300](https://notes-media.kthiha.com/Breadth-First-Search-(BFS)/c9e0e9a71decd9e15a0fd36155bf627e.png)

---
A [[Breadth-First Search (BFS)|BFS]] can give the following information about a [[graph]].
- The shortest path from $v$ to any other vertex $u$.
  We denote the distance between the nodes as $d(v)$.
- Whether the [[graph]] is connected.
- The number of connected components.
- Constructs a `spanning tree` that visits every node connected to the starting node.
  **Note**: Because a [[Breadth-First Search (BFS)|BFS]] follows from an adjacency list, the spanning tree is not unique.
  **Note**: [[Breadth-First Search (BFS)|BFS]] is slow and takes a lot of work to get the solution.

---
## Implementing BFS
We can use a [[Priority Queue|queue(FIFO)]] to implement a [[Breadth-First Search (BFS)|BFS]] given an adjacency list representation of a [[graph]].

A queue has the following properties:
- `enqueue(Q,V)`
- `dequeue(Q)`
- `isEmpty(Q)`

Furthermore, we will need to store the following information for each node $v$:
- The current node $u$ and its state (visited, not visited, finished)
- The predecessor $p[u]$
- The distance from $u$ to $v$
- The order of discovery

---
## Complexity
Since each node is enqueued at most once, the adjacency list of each node is examined at most once.

Therefore, the total running time of [[Breadth-First Search (BFS)|BFS]] is $O(m+n)$ or linear in the size of the adjacency list.

**Note**: 
- Each node is enqueued when it is not visited, at which point it is marked as visited.
- [[Breadth-First Search (BFS)|BFS]] will only visit the nodes that are reachable from $V$.
- If the [[graph]] is connected(in the undirected case) or strongly-connected(in the directed case), then this will all be vertices.
- If not, then we may have to call [[Breadth-First Search (BFS)|BFS]] multiple times in order to see the whole [[graph]].

---
## See Also
- [[Graph]]
- [[Priority Queue]]
