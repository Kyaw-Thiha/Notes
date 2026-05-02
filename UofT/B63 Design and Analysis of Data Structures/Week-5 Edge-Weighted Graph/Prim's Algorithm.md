# Prim's Algorithm
[[Prim's algorithm]] builds an [[Minimum Spanning Tree (MST)|MST]] by adding the smallest edge at each step, starting from a random node.
![Prim's Algorithm|300](https://i.ytimg.com/vi/xthRL0lcx2w/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLDTvrg1fOPERXKgj6f1SQiHjzM6rA)
It can be thought of as a [[Greedy Policy|greedy algorithm]] that choose the locally optimal solution at each step, and hope to find the global optimum.

---
## Algorithm
**Idea**: Find an [[Minimum Spanning Tree (MST)|MST]] by doing something similar to [[Breadth-First Search (BFS)|BFS]].
- Use a [[priority queue]] to store weights.
- The algorithm grows a tree $T$ one edge at a time.
- The [[Priority Queue|priority]] of vertex $V$ is the smallest edge weight between $v$ and $T$ so far. We use $\infty$ if there is no such weight.

![image|300](https://notes-media.kthiha.com/Prim's-Algorithm/d87a246c39fdb8042123e67ffb0ae66a.png)

> **Algorithm Summary**
> 
> With [[Prim's Algorithm]], you choose any vertex and transverse the [[graph]] by taking the path with the least weight.
> Furthermore if a vertex can have a smaller [[Priority Queue|priority]], you must update the [[Heap|min-heap]] to show the smallest possible priority it can have.
> Finally, we remove the vertex from the [[Heap|min-heap]] when we have visited it.

---
## Example
Consider the [[graph]] below.
![image|300](https://notes-media.kthiha.com/Prim's-Algorithm/dd537f604a56a90c1f0c07fd7277b69e.png)

![image|300](https://notes-media.kthiha.com/Prim's-Algorithm/957d1aaae0c6e7cd62028b5488e68f38.png)
![image|300](https://notes-media.kthiha.com/Prim's-Algorithm/00c2bc7d308f88b5195ccf849de1b734.png)
![image|300](https://notes-media.kthiha.com/Prim's-Algorithm/e718a4fb208443248f0816264b6b56d7.png)
![image|300](https://notes-media.kthiha.com/Prim's-Algorithm/217cfb974ef284da38df8d1aae400cbd.png)
![image|300](https://notes-media.kthiha.com/Prim's-Algorithm/b84edd73fdf4f71ae78642d41671efdd.png)

---
## Complexity
- Every vertex enters and leaves the [[Heap|min-heap]] once, so it takes $\Theta(\log n)$ per vertex.
  In total, it takes $\Theta(n \log n)$.
- Decreasing all vertices [[Priority Queue|priorities]] takes $O(m \log n)$ time.
- In total, it takes $O((m+n) \log n)$ time.
- Note that for adjacency matrix representation, it is $O(m^{2})$.

---
## Proof of Correctness
### Cut Property
> Let $S$ be a nontrivial subset of $V$ in $G$. $(S \neq \emptyset \text{ and } S\neq V)$
> If $(u,v)$ is the lowest-cost edge crossing $(S,\  V-S)$, then $(u,v)$ is in every [[Minimum Spanning Tree (MST)|MST]] of $G$.

**Proof**:
Proof by contradiction.
- Suppose there exists [[Minimum Spanning Tree (MST)|MST]] $T$ that does not contain $(u,v)$.
- Consider the sets $S$ and $V-S$.
  There must be a path from $u$ to $v$.
- On this path, there must exists an edge $e$ that crosses between $S$ and $V-S$.
- Since $(u,v)$ is the least weighted edge crossing between $S$ and $V-S$, swapping $e$ with $(u,v)$ will reduce the weight of $T$
- $\therefore$ $T$ is not an [[Minimum Spanning Tree (MST)|MST]].

---
### Proving Prim's Algorithm
- Consider an [[Minimum Spanning Tree (MST)|optimal MST]] $O$ and [[Prim's Algorithm]] tree $T$.
- Order the edges of $T$ and $O$ in the order they were selected.
- Consider the first edge $e=(u,v)$ in that ordering that is in $T$ and not in $O$.
- At the stage of [[Prim's Algorithm]] when $e$ was added, there was a set $S$ of vertices $s.t.$ $u \in S$ and $v \in V-S$.
- If the [[Weighted Graph|edge weights]] are unique, then by the [[#Cut Property]], $e$ must belong to $O$.
- If the [[Weighted Graph|edge weights]] are not unique and if $e \notin O$, that means that there exists a path $p$ from $u$ to $v$ $s.t.$ an edge $e' = (x,y)$ exists on $p$ and $x \in S$ and $y \in V-S$.
- If $w(e) = w(e')$, then we can swap $e$ with $e'$ and the [[spanning tree]] will still be optimal.
- It is impossible for $w(e')$ to be less than $w(e)$ or [[Prim's Algorithm]] would have chosen it.
- If $w(e) < w(e')$, then swapping $e'$ with $e$ reduces the weight of $O$, which is a contradiction.

---
## See Also
- [Good blog post by Olivia](https://oliviagallucci.com/optimization-via-kruskals-prims-minimum-spanning-trees-msts/)
- [Visualization Video](https://youtube.com/shorts/SUFLBSARh0g?si=JbXXSBUPZ0HQ1MQG)
- [Simulator](https://www.cs.usfca.edu/~galles/visualization/Prim.html)
- [[Minimum Spanning Tree (MST)]]
- [[Kruskal's Algorithm]]