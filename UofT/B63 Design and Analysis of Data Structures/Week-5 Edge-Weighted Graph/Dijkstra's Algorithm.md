# Dijkstra's Algorithm
[[Dijkstra's algorithm]] finds the shortest path between $2$ nodes.
![Dijkstra's Algorithm|300](https://blog.aos.sh/img/DAGIF.gif)
Note that it is very similar to [[Prim's algorithm]].

---
## Example
Consider the graph below.
![image|300](https://notes-media.kthiha.com/Dijkstra's-Algorithm/9e942990f6b729a94e9d2d2984a42d8c.png)

![image|300](https://notes-media.kthiha.com/Dijkstra's-Algorithm/1184d1f4f00c0e139aca786fd9a856db.png)
![image|300](https://notes-media.kthiha.com/Dijkstra's-Algorithm/f049d367a21092df7b0dc66a2d5afbe0.png)
![image|300](https://notes-media.kthiha.com/Dijkstra's-Algorithm/cb928ec952de1ca5f6fe88fb86af3b55.png)
![image|300](https://notes-media.kthiha.com/Dijkstra's-Algorithm/5c9e06c39b3073c3853665afab29de42.png)
![image|300](https://notes-media.kthiha.com/Dijkstra's-Algorithm/7e4836fea35d5312b271412f7618f19a.png)
![image|300](https://notes-media.kthiha.com/Dijkstra's-Algorithm/cba97c664eb4f9b7f21ce7e80f77f95a.png)

---
## Algorithm
1. We add our start vertex $S$ to the set of reached vertices $S$ and give it a distance $d[S]=0$.
   This creates a distance tree rooted at $S$.
2. At each stage, we consider the next closest vertex to $S$ from vertices not in $S$, or alternatively, the vertex with the next shortest path to $S$.
3. We can use a [[priority queue]] to determine the shortest path by considering each neighbour $u$ of $v$ $s.t.$ $u \notin S$ whenever a new vertex $v$ is added to $S$. 
   If we get a smaller [[Priority Queue|priority]] for $u$, we update $u$'s priority to the new priority.

---
## Proof
- Let $T$ be the distance tree constructed by [[Dijkstra's Algorithm]], starting at $S$.
- Let $O_{S}$ be an optimal distance tree rooted at $S$.
- Order the edges $\langle \  e_{1}, e_{2}, \dots, e_{m} \ \rangle$ according to how they are added to $T_{S}$.
- Consider the first edge $e_{i}=(u,v)$ $s.t.$ $e_{i} \in T_{S}$ and $e_{i} \notin O_{S}$.
- Then, $e_{1}, \dots, e_{i-1} \in T_{S}$.
  Let $S$ be the set of vertices added so far. $\text{I.e}: \langle \ e_{1}, \dots, e_{i-1} \ \rangle$.
- Each node in $S$ has a min path distance to $S$.
- Since $(u,v) \notin O_{S}$, there must be a shorter path $p$ from $S$ to $v$.
- Consider the edge $e_{j}=(x,y)$, $j>i$ on $p$ that has one endpoint in $S$ and one in $V-S$.
	- **Case-1**: $y\neq V$
	  $d_{O}[y] < d_{T}[V]$. This is a contradiction since [[Dijkstra's algorithm]] would have chosen it.
	- **Case-2**:
		- **Case-2A**: If $y=V$ and $d_{O}[y] < d_{T}[V]$, then this is a contradiction because [[Dijkstra's algorithm]] would have chosen it.
		- **Case-2B**: If $y=V$ and $d_{O}[y] = d_{T}[V]$, we can swap $(x,y)$ with $(u,v)$ and $O_{S}$ is closer to $T_{S}$.

---
## See Also
- [[Weighted Graph]]
- [[Prim's Algorithm]]