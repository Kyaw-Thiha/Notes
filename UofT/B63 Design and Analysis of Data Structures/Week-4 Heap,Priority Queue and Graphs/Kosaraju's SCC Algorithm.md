# Kosaraju's SCC Algorithm
- [[Depth-First Search (DFS)|DFS]] on $G$.
  Visit all the vertices, note the finish times and accumulate vertices in reverse finishing order.
- Compute the adjacency list of $G^{T}$.
- [[Depth-First Search (DFS)|DFS]] on $G^{T}$, using above order to pick start/restart vertices.
- Each tree found has the vertices of one [[Strongly Connected Components (SCC)|SCC]].
  In total, this takes $O(|V| + |E|)$ time.

---
## Example
Consider the [[graph]] below.
![image|200](https://notes-media.kthiha.com/Kosaraju's-SCC-Algorithm/7012c696c28d0999d0587d56dbbbe8b9.png)

![image|300](https://notes-media.kthiha.com/Kosaraju's-SCC-Algorithm/27b9ae9f0ef2b346d35d9aeaeb94762d.png)
![image|300](https://notes-media.kthiha.com/Kosaraju's-SCC-Algorithm/6b60ce63f01b96c01f3e09aa8e62dd6e.png)
![image|300](https://notes-media.kthiha.com/Kosaraju's-SCC-Algorithm/6676bfdd1246992c5024ee404d07f520.png)
![image|300](https://notes-media.kthiha.com/Kosaraju's-SCC-Algorithm/87dd4bd88ea4d896966a37ec91310fb0.png)
![image|300](https://notes-media.kthiha.com/Kosaraju's-SCC-Algorithm/622de83f7fe69a2376bd9d6e537e4996.png)
![image|300](https://notes-media.kthiha.com/Kosaraju's-SCC-Algorithm/c32af297b5e1c02c6be9689432787631.png)
![image|300](https://notes-media.kthiha.com/Kosaraju's-SCC-Algorithm/f0d67bb8b7fff884e7b4711013356965.png)
![image|300](https://notes-media.kthiha.com/Kosaraju's-SCC-Algorithm/5d3b4238845bfda2c1c8d228ac153005.png)

---
## Proof of Kosaraju's Algorithm
**Notation**:
- Let's denote $f(v)$ as the time of which vertex $v$ is finished.
- $f(u) < f(v)$ means $u$ is finished before $v$.
- Let $C$ be the [[Strongly Connected Components (SCC)|SCC]].
  We define $f(C)$ to be the time at which the last node in $C$ finishes. Formally, $f(C) = \max_{v \in C} f(v)$.

---
> **Lemma**:
> If $S$ is the first node in [[Strongly Connected Components (SCC)|SCC]] $C$ visited by [[Depth-First Search (DFS)|DFS]], then $f(c) = f(s)$.

**Proof**:
- Since $S$ is the first node in [[Strongly Connected Components (SCC)|SCC]] $C$ visited by [[Depth-First Search (DFS)|DFS]], all vertices in $C$ are not finished. 
- Furthermore since $C$ is a [[Strongly Connected Components (SCC)|SCC]], every vertex in $C$ is reachable from $S$. That means that there is a path from $S$ to every vertex in $C$.
- Thus, every node will be finished when [[Depth-First Search (DFS)|DFS]] returns.
- Since the last step of the [[Depth-First Search (DFS)|DFS]] is to finish $S$, this means that $S$ is finished only after all other vertices are finished.
- Therefore, $f(S) > f(V)$ for any $V \in C$.
- By the definition of $f(C) = \max_{V \in C} f(V)$, $f(C)=f(S)$.

---
> **Theorem**:
> Suppose we run [[Depth-First Search (DFS)|DFS]] starting at each node in $G$.
> Let $C_{1}$ and $C_{2}$ be the [[Strongly Connected Components (SCC)|SCCs]] in $G$.
> 
> If $(u,v)$ is an edge in $G$ where $u \in C_{1}$ and $v \in C_{2}$ and $f(u) < f(v)$, 
> then $f(C_{1}) < f(C_{2})$.


**Proof**:
- Let $x_{1}$ and $x_{2}$ be the first vertices [[Depth-First Search (DFS)|DFS]] visits in $C_{1}$ and $C_{2}$.
- By our **lemma**, $f(c_{1})=f(x_{1})$ and $f(c_{2})=f(x_{2})$.
  Therefore, we will show $f(x_{2}) < f(x_{1})$.
- **Note**: $x_{2}$ is reachable from $x_{1}$, because there is a path from $x_{1}$ to $u$ in $C_{1}$ across $(u,v)$ and a path from $v$ to $x_{2}$ in $C_{2}$.
- However, $x_{1}$ is not reachable from $x_{2}$ since $x_{1}$ and $x_{2}$ would be strongly connected, contradicting that they belong in different [[Strongly Connected Components (SCC)|SCCs]].
- We have $2$ cases:
	1. $\text{DFS}(x_{2})$ is called before $\text{DFS}(x_{1})$:
	  Since $x_{1}$ is not reachable from $x_{2}$, $x_{2}$ will finish before $x_{1}$.
	  $\therefore$ $f(x_{2}) < f(x_{1})$, as wanted.
	2. $\text{DFS}(x_{1})$ is called before $\text{DFS}(x_{2})$.
		- When $\text{DFS}(x_{1})$ is called, all nodes in $C_{1}$ and $C_{2}$ have not been visited, so there is a [[Depth-First Search (DFS)|DFS]] path from $x_{1}$ to $x_{2}$.
		- When $\text{DFS}(x_{1})$ returns, $x_{2}$ will be finished.
		- Since $x_{1}$ will be finished just before $\text{DFS}(x_{1})$ returns, this means that $x_{1}$ finished before $x_{1}$, so $f(x_{2}) < f(x_{1})$.

---
> **Corollary**: 
> Let $C_{1}$ and $C_{2}$ be distinct [[Strongly Connected Components (SCC)|SCCs]] in $G=(V,E)$.
> Suppose there is an edge $(u,v)$ in $E^{T}$ where $v \in C_{1}$ and $v \in C_{2}$, and $f(u) < f(v)$.
> 
> Then, $f(C_{1}) < f(C_{2})$.

---
> **Corollary**:
> Let $C_{1}$ and $C_{2}$ be the two distinct [[Strongly Connected Components (SCC)|SCCs]] in $G = (V,E)$.
> If $f(C_{1}) > f(C_{2})$, then there cannot be an edge from $C_{1}$ to $C_{2}$ in $G^{T}$.

Consider this:
Since we know that $f(C_{1}) > f(C_{2})$, then there is an edge from $C_{1}$ to $C_{2}$. However in $G^{T}$, that edge is reversed, so there is no longer an edge from $C_{1}$ to $C_{2}$.
![image|300](https://notes-media.kthiha.com/Kosaraju's-SCC-Algorithm/54190840f4f33a0df58f747c86be4e62.png)

So if we start the [[Depth-First Search (DFS)|DFS]] on $G^{T}$ at $C_{1}$, because there is no edge from $C_{1}$ to $C_{2}$, the [[Depth-First Search (DFS)|DFS]] will only visit the vertices from $C_{1}$ and it will return a [[Depth-First Search (DFS)|DFS]] tree that contains only vertices from $C_{1}$.

Then, when you do [[Depth-First Search (DFS)|DFS]] on $C_{2}$, even though there is an edge from $C_{2}$ to $C_{1}$, [[Depth-First Search (DFS)|DFS]] will only visit the vertices in $C_{2}$ because we already finished $C_{1}$.

We continue for all remaining [[Strongly Connected Components (SCC)|SCCs]].

---
**Proof**:
- Edge $(u,v) \in E^{T}$ implies $(v,u) \in E$.
- Since [[Strongly Connected Components (SCC)|SCCs]] of $G$ and $G^{T}$ are the same, $f(C_{2}) > f(C_{1})$.
- This completes the proof.

---
## See Also
- [[Strongly Connected Components (SCC)]]
- [[Graph]]
- [[Depth-First Search (DFS)]]
- [[Breadth-First Search (BFS)]]
