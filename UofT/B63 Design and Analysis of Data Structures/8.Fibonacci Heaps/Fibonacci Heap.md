# Fibonacci Heap
[[Fibonacci Heap|Fibonacci heaps]] are a forest of [[Heap|heap-ordered]] [[AVL Tree|trees]]. The parent's priority is always less than or equal to its children priority.

---
### The Root List
- The roots are stored in a circular doubly-linked list. 
  This list is called `The Root List`.
- There is a pointer to the min root.
- The siblings are also stored in a circular doubly-linked list.
  However, the parent only knows one arbitrary child.


---
### Node
Each node has the following:
- `key`: The node's priority.
- `left,right`: Pointer to the left and right sibling.
- `parent`: Pointer to the parent.
- `child`: Pointer to one child.
- `degree`: The number of children.
- `mark`: A boolean. Set to false, but if a node loses a child, it is set to true. This is important during `decrease-priority`.
- `min`: The whole heap has a pointer to the min cost.

---
### Example
![image|300](https://notes-media.kthiha.com/Fibonacci-Heap/70e5976e25a8bfb39826bf16466531f1.png)
Note that 
- the dashed line represent circular doubly-linked list
- red represents the root list

---
### Definition
- We define $\text{H} \cdot n$ to be the number of nodes in $\text{H}$.
- We define $\text{deg}(x)$ to be the number of children in $x$'s child list. $\text{deg}(3) = 3$
---
## Operations
- `make_heap()`: Creates a new empty [[Heap|heap]].
- `insert(H,x)`: Insert $x$ to $\text{H}$.
- `min(H)`: Return a pointer to the min key in $\text{H}$.
- `extract_min(H)`: Deletes the element from $\text{H}$ whose key is min and returns a pointer to the key.
- `union(H1,H2)`: Creates and returns a new heap that contains the elements from $\text{H}_{1}$ to $\text{H}_{2}$.
- `decrease_key(H,X,key)`: Assigns to element $x$ in heap $H$ the new key value `key`.
- `delete(H,x)`: Delete key $x$ from $H$.

---
### Comparism to Binary Heap
![image|300](https://notes-media.kthiha.com/Fibonacci-Heap/5ee56cfe98a607d03406abbc3b261f7a.png)

If [[Prim's Algorithm]] used [[Fibonacci Heap]], 
- $|V|$ extract-mins: $O(|V| \ lg(|V|))$ time
- up to $|E|$ dec-pris: $O(|E|)$ time
- Total: $O(|V| \ \lg(|V| + |E|))$ time

---
### Insert
- `insert(k)`:
	- Insert the new node in the root list.
	- $\text{key} = k$
	- $\text{mark = false}$
	- Change $\text{min}$ if $k < \text{min.key}$
- Takes $\theta(1)$ time.
- Example:
![image|300](https://notes-media.kthiha.com/Fibonacci-Heap/91148516304d9be339e15d34a03aab5e.png)

---
### Union
- `union(H1,H2)`:
	- Concatenate the $2$ root lists.
	- $H_{1}.\text{min}$ and $H_{2}.\text{min}$ compete the new $\text{min}$.
- Takes $\theta(1)$ time.
- Example:
![image|300](https://notes-media.kthiha.com/Fibonacci-Heap/94b6e72488583d3790708ce3d98a41e3.png)

---
### Extract-Min
- The `min key` is already pointed to by $\text{H.min}$.
- We remove the min root, and promote its children to the root list.
- Now, $\text{H.min}$ may point to any node on the root list.
  (May not be the correct node).
- We consolidate the [[Fibonacci Heap|fib heap]]. 
  I.e: We want to end up with a root list with nodes of unique degrees.

[[Extract-Min Fibonacci Heap|Read More]]

---
### Decrease Key
The pseudocode of `decrease_key(H,x,k)` is
![image|300](https://notes-media.kthiha.com/Decrease-Key-Fibonacci-Heap/47687863401bc982872b60c7d73c346c.png)
![image|300](https://notes-media.kthiha.com/Decrease-Key-Fibonacci-Heap/1a0669861c0864d1a0360035946a8242.png)
![image|300](https://notes-media.kthiha.com/Decrease-Key-Fibonacci-Heap/9812c39138b328b7f9c38e4ab2b3dcba.png)

[[Decrease-Key Fibonacci Heap|Read More]]

---
## Complexity
Lets look at the actual [[Time Complexity|worst case costs]].
- `insert`: $O(1)$
- `extract_min`: 
	- Removing the `min_costs`: $O(1)$
	- Moving the removed node's children up to the root list takes $O(d(v))$ where $v$ is the min.
	- When we consolidate, each root can be the child of another root at most once. We do at most $O(trees(H))$ merges.
	- Find the new `min` takes $O(d(n))$ where $d(n)$ is the max degree for all root nodes $n$.
	- Total: $O(trees(H) + d(n))$
- `decrease_priority`:
	- Update the priority of the key: $O(1)$.
	  Note:
		- If the heap order is not violated, then we're done.
		- However, if the heap order is violated, then we cut the node and insert it in the root list.
		- This takes $O(1)$.
		- So in total, updating the priority of the key takes $O(1)$ time.
	- Cascading cut takes $O(\text{\# of cuts})$.
	- In total, it costs $O(\text{\# cuts + 1})$.
	  Note: $O(\text{\# cuts + 1}) \leq O(\text{marks(H) + 1})$.

---
### Amortized Analysis
- `insert`: Amortized cost $O(1)$
- `extract_min`: Amortized cost $O(\log n)$
- `decrease_priority`: Amortized cost $O(1)$

Read more about the [[Amortized Analysis|amortized analysis]] [[Amortized Analysis of Fibonacci Heap|here]].

---
## See Also
- [[Fibonacci Heap]]
- [[Extract-Min Fibonacci Heap]]
- [[Decrease-Key Fibonacci Heap]]
- [[Amortized Analysis of Fibonacci Heap]]
- [[Heap]]