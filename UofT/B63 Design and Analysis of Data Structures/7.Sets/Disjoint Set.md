# Disjoint Set
[[Disjoint Set|Disjoint sets]] are a collection of sets with no elements in common. $\text{I.e:}$ Disjoint sets are sets whose intersection is the empty set.

---
## Notes
- A [[Disjoint Set|disjoint-set data structure]] maintains a collection $S= \{ S_{1}, S_{2}, \dots, S_{k} \}$ of disjoint dynamic sets.
- We can identify each set by a representative, which is a member of the set.
- $\text{E.g:}$ $\{ 1,2,3 \}$, $\{ 4,5,6 \}$, and $\{ 7,8,9 \}$ are [[Disjoint Set|disjoint sets]].

---
## Operations
- `make-set(x)`: Creates a new singleton set containing $x$.
- `find(x)`: Find the set $x$ is in.
- `union(S, S')`: Merges sets $S$ and $S'$
  `union(x, y)`: Merges $x$'s owner and $y$'s owner.
  We assume that they are in $2$ different sets.

---
## Set
A set is anything that supports the following:
- $\text{find}(x) = \text{find}(y)$
- $\text{union}(\text{find}(x), \text{union}(y))$

---
## Linked List
> We can represent each [[Disjoint Set|set]] using one [[#Linked List|linked list]] per [[Disjoint Set|set]].

Each element in the [[#Linked List|linked list]] will have a pointer to the rep of the set, which is usually the head of the list.

---
### Amortized Analysis
Given a sequence of $m$ operations, including $n$ `make-sets`, we can calculate the [[Amortized Analysis|amortized time]] using [[Aggregate Method (Amortized Analysis)|aggregate method]] from the following:
- Each time we union $2$ [[Disjoint Set|disjoint sets]], we have to update the rep field of all the new elements.
- To save time, we always add the elements from the smaller [[#Linked List|linked list]] to the bigger [[#Linked List|linked list]].
- Each time $x.rep$ is updated, $x$ lands in a [[#Linked List|list]] at least twice as long as before.
- Since the max length of the [[#Linked List|linked list]] is $n$ at most, we will update each rep field $\log n$ times. Since there are $n$ owner fields, we have at most $n \log(n)$ updates in total.
- The other steps cost $O(1)$ per operation, so $O(m)$ time.
- Altogether, it is $O(m + n\log(n))$ or $O(m \log n)$.
  This gives an [[Amortized Analysis|amortized time]] of $O(\log n)$.

---
### Pseudocode for Union

![image|300](https://notes-media.kthiha.com/Disjoint-Set/084c0863e1b4a71df08c512ad31399e6.png)

---
## Tree
Each set can be represented as a [[#Tree|directed tree]] where directed edges go from children to parent, and elements are nodes.

- We use the root to be the representative of the [[Disjoint Set|set]].
  $\text{I.e}:$ $\text{find}(x)$ returns the root of $x$
- We will store a [[Augmented AVL Tree|rank field]] at every node.
  The rank represents the max height at that node.
![image|300](https://notes-media.kthiha.com/Disjoint-Set/cae498572dd329e58fd2c79b93d15222.png)
- Initially, sets begin as single element trees.
  $\text{I.e}:$
```python
make-set(a):
	a.rep = a
	a.rank = 0
```
- If we are unionizing $2$ [[Disjoint Set|disjoint sets]] using the [[#Tree|tree]], we always attach the tree with the lower rank to the tree with the higher rank. This is done to save time.
- If the ranks are equal, we update the new root's rank by adding $1$.

---
### Pseudocode for Union
![image|400](https://notes-media.kthiha.com/Disjoint-Set/8c6a202f8b3e1e291234fc92f79a324b.png)

> **Note**: $m$ calls to {`make-set`, `union`, `find`} becomes up to $3m$ calls to {`make-set`, `union`, `find`}.

---
### Path Compression
To find the root from any element using the [[#Tree|tree]], we can start from the element and go up to the root. 

However, a better way would be to use a **path compression**. We can update every node along the path to the root directly.

![image|300](https://notes-media.kthiha.com/Disjoint-Set/e18764aa69377871871f14a3a285c954.png)

Think of this as "lazy-updating" the pointer to root.

---
## Complexity
- The best [[Disjoint Set|disjoint set]] implementation is [[#Tree|trees]] using union with [[Augmented AVL Tree|rank]] and find with [[#Path Compression|path compression]].
- The [[Time Complexity|worst-case time]] for a seq of $m$ operations, where there are $n$ `make-sets` is $O(m \log ^{*} n)$

> **Note**: $\log ^{*}$ is the number of times that you need to apply $\log$ to $n$ until the answer is less than $1$.

$$
\begin{align}
\text{E.g: } \quad &\text{Let } n=40 \\[6pt]
&5 < \log(40) < 6 \\[6pt]
&2 < \log(\log 40) < 3 \\[6pt]
&1 < \log(\log( \log 40)) < 2 \\[6pt]
&0 < \log(\log(\log(\log 40))) < 1 \\[6pt]
\end{align}
$$

> **Note**: 
> Higher bases give smaller $\log^{*}$, also called **iterated logs**.

---
#### Using inverse Ackermann's Function
> The only commonly used function in [[Time Complexity|complexity theory]] that grows more slowly is the [[Ackermann's Function|inverse Ackermann's Function]].

- The [[Amortized Analysis|amortized times]] of `link` and `find` are $O(\alpha(n))$ if there are $n$ elements.
- $\alpha(|V|) \in O(\log |V|)$
- Recall that [[Kruskal's Algorithm]] used [[Disjoint Set|disjoint sets]].
  Hence, it had $O(|E| \ \log |E| + |E| \ \alpha(|V|))$ [[Time Complexity|time]].
  Since $O(|E| \ \log|E|) \in O(|E| \ \log|V|)$, [[Kruskal's Algorithm]] takes $O(m\log n)$ time.

---
## See Also
- [[Ackermann's Function]]
- [[Augmented AVL Tree]]
- [[Kruskal's Algorithm]]
- [[Amortized Analysis]]
- [[Time Complexity]]