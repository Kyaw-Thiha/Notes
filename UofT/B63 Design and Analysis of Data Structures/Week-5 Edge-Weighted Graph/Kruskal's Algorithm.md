# Kruskal's Algorithm
[[Kruskal's algorithm]] builds the [[Minimum Spanning Tree (MST)|MST]] by iteratively adding the smallest edge that does not form a cycle until it includes all vertices.

![Kruskal's Algorithm|300](https://i0.wp.com/oliviagallucci.com/wp-content/uploads/2024/01/KruskalsAlgorithm.gif?resize=549%2C379&ssl=1)

---
## Example
Consider the graph below.
![image|300](https://notes-media.kthiha.com/Kruskal's-Algorithm/a3b7e4cc12fff14fe7115119d2412613.png)

1. Grab the edges with weight of $2$.
![image|200](https://notes-media.kthiha.com/Kruskal's-Algorithm/04b5cfa1914fd39dc3b86a3f9a2dc135.png)
2. Grab the edges with weight of $3$.
![image|200](https://notes-media.kthiha.com/Kruskal's-Algorithm/831703ab942b1f1a702e0a93141d4784.png)
3. Because grabbing the edges of weight $4$, $5$ and $6$ would create a cycle, we skip those. We instead grab the edge of weight $7$. We are now done.
![image|300](https://notes-media.kthiha.com/Kruskal's-Algorithm/ffdf1c641a9a86bd7391ef82c339c718.png)

We know we are done when:
- we have visited each node.
- we have $n-1$ edges.

---
**Note**: If we have a tree, then we must have $n-1$ edges.
However if we have $n-1$ edges, it does NOT mean we have a tree.
![image|300](https://notes-media.kthiha.com/Kruskal's-Algorithm/31cd6ca60b50bd0b29bfebd593b1fd4e.png)

---
### Comparism to Prim's Algorithm

Compared to [[Prim's algorithm]], [[Kruskal's algorithm]] is more well-suited for sparse [[Graph|graphs]].

---
## Implementation
- Store the edges sorted in non-decreasing weight in a [[Priority Queue|min-priority queue]].
- To add edges and to make sure that no [[Graph|cycle]] is induced, we can use linked lists.
  Think of it as joining together clusters(subtrees) of connected vertices. Each cluster is a linked list.
- To make sure that adding an edge does not induce a [[Graph|cycle]], check to see if the vertex that the new edge points to is in the linked list or not.
  - If it is, then adding the edge would create a cycle.
  - If it isn't, then adding the edge wouldn't create a cycle.
- Merging $2$ clusters is merging $2$ linked list.
  However, a lot of vertices need their cluster pointers updated. If we move the smaller linked list to the bigger one, updating each vertex takes $O(\log n)$ time at most. 
  In total, it takes $O(n \log n)$.

---
### Time Complexity
- Building [[Priority Queue|priority queue]] and removing edges takes $O(m \log n)$.
- Updating all the vertices takes $O(n \log n)$.
- The rest is $\Theta(1)$ per edge or vertex.
- In total, it takes $O(m\log n + n \log n)$.
  But because $m \leq n^{2}$, $\log m \in O(\log n)$.

$\therefore$ The [[time complexity]] is $O((m+n) \log n)$.

---
## Proof of Correctness
We can prove [[Kruskal's algorithm]] is correct using proof by contradiction.

**Proof**:
- We order the edges in non-decreasing order of weights.
  $\text{I.e:}$ $w_{1} \leq w_{2} \leq w_{3} \leq \dots \leq w_{n}$
- Let $K$ be the [[spanning tree]] returned by [[Kruskal's Algorithm]].
- Let $O$ be an [[Minimum Spanning Tree (MST)|optimal MST]] $s.t.$ $O$'s weight is less than $k$'s weight. This means that $K$ is not optimal.
- Let $e_{i} = (u,v)$ be the first edge in our ordering that is not in both $K$ and $O$.
  **Note**: 
	- $e_{i} \in K$ and $e_{i} \notin O$ because $K$ only omits edges if they create a cycle. 
	- Since $O$ is a [[Minimum Spanning Tree (MST)|MST]], it can't have any cycles.
	- Therefore, it is not possible for $e_{i} \notin K$ and $e_{i} \in O$.
- Since $O$ is connected, there must exist a unique path $p$ from $u$ to $v$ and an edge $e'$ on $p$ that is not in $K$.
- Since $K$ didn't choose $e'$ but had the option to.
  This means $w(e') \geq w_{i}$.
	- **Case-1**: $w(e') = w_{i}$
	  Then, we can switch $e_{i}$ and $e'$ and $O$ still has the same weight as before but is more similar to $K$.
	  Repeat this argument until either $\text{case-2}$ or the $2$ trees are the same and $K$ is optimal.
	- **Case-2**: $w(e') > w_{i}$
	  Now, consider a tree $O'$ constructed by removing $e'$ from $O$ and adding $e_{i}$. Now, $O'$ has less weight than $O$ contradicting that $K$ and $O$ differ.
- Therefore $K$ must be optimal.

---
## See Also
- [Good blog post by Olivia](https://oliviagallucci.com/optimization-via-kruskals-prims-minimum-spanning-trees-msts/)
- [[Minimum Spanning Tree (MST)]]
- [[Prim's Algorithm]]