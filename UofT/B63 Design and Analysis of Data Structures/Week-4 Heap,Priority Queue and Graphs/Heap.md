# Heap
A [[heap]] is one way to store a [[priority queue]].

A [[heap]] is 
- a binary tree
- every level $i$ has $2^{i}$ nodes, except for the bottom level
- The bottom fills in left to right
- At each node, its priority is greater than or equal to both of its children's priorities

---
### Insert
Suppose we have this heap.
![image|300](https://notes-media.kthiha.com/Heap/072d82ec22e26c760ade2053e03fc037.png)

If we insert a key with priority $15$, then this happens:
![image|300](https://notes-media.kthiha.com/Heap/2861d8a99b6da7e98521f84e485b8058.png)
![image|300](https://notes-media.kthiha.com/Heap/625a784fe42f24b8f3e0b3b47cb3ea9e.png)
![image|300](https://notes-media.kthiha.com/Heap/47cb0f5083660e104db5d088ab5fd52d.png)

#### Pseudocode
1. Create a new leaf at the bottom level, leftmost open location.
   This maintains the requirement that the tree is nearly-complete.
2. Assign the priority and job in the new node.
   Let $v=\text{new node}$.
3. We percolate $v$ up.
   while $v$ has parent with smaller priority:
	- swap them
	- $v = v.\text{parent}$

The worst case time is $\Theta(\text{height})$.
The height is $[\log n] + 1$, so the worst case time is $\Theta(\log n)$.

---
### Extract-Max
Suppose we have this heap:
![image|300](https://notes-media.kthiha.com/Heap/28862e8334fd0baee1773dc294312041.png)

If we [[#Extract-Max|extract-max]], then this happens
![image|300](https://notes-media.kthiha.com/Heap/aaa84a97d29e8187adcc9fa3c582edf5.png)
![image|300](https://notes-media.kthiha.com/Heap/2344f2fb70ad656e30ad0ae7dd97d740.png)
![image|300](https://notes-media.kthiha.com/Heap/2335d7d8136f967e8d4e7151d5c72f57.png)

#### Pseudocode
1. Replace the root with the bottom level, rightmost item.
   This keeps the tree nearly-complete.
2. $v=root$.
3. while $v$ has larger child:
	- swap with the largest child
	- $v = \text{that child node}$

The worst case time is $\Theta(\text{height})$.
The height is $[\log n] + 1$, so the worst case time is $\Theta(\log n)$.

---
## Heap Height
Let $n$ be number of nodes a binary tree of height $h$ can have.
At most, the binary tree can have $2^{h}-1$ nodes.
At the very least, the binary tree can have $2^{h-1}$ nodes.

![image|200](https://notes-media.kthiha.com/Heap/d24cc971e838613c75735800ef516cf9.png)

---
## Implementing a Heap
We can use an array to implement a heap.
- The start index is $1$.
- Left child of node $i$ is $2i$.
- Right child of node $i$ is $2i+1$.
- Parent of node $i$ is $\left[ \frac{1}{2} \right]$.

![image|300](https://notes-media.kthiha.com/Heap/e789ee64f021092f3949093cda70bd3d.png)

**Insertion**
- To insert an item into a heap with an array, we append the new item into the end of the array.
- Next, we update the heap size if necessary.
- And then we percolate the new node upwards.

**Extract-Max**
- To extract max, we read and replace the item at index $1$ with the item at the very end (last index) of the heap.
- Next, decrement the heap size if necessary.
- And then, heapify starting at index $1$.

**Increasing Priority Value**
- If we know where a node is in a heap stored in array $A$, we can increase the priority value in $\Theta(\log n)$ time.
- We then set the new $p$ value and then percolate the node upwards.

Consider the following pseudocode.
![image|300](https://notes-media.kthiha.com/Heap/67a0070321484eef7aa1988a4a4b8ec8.png)

- The logic is that if $P$ is less than or equal to the current priority, we do nothing.
- If $p$ is greater than current priority, we set priority to $p$ and we percolate up.
- We compare the priority of the element to the priority of the parent. Then, we swap if the parent's priority is smaller than the node's priority.
- We continue this until we reach the root or until we get to a node who has a higher priority.

---
## Building Heaps
Given an array $A$ of elements with priorities, whose only empty slots are at the far right, how can we turn $A$ into a heap?

1. Sort $A$ from highest priority element to lowest.
   $\Theta(n \log n)$
2. Create a new array $B$ that represents a heap and go through every element of $A$ and insert it into $B$.
   $\Theta(n \log n)$
3. Use the fact that each subtree of a heap is a heap.
   We start with the smallest subtrees, turning them into heaps using heapify.
   Then, we work up the tree.
![image|300](https://notes-media.kthiha.com/Heap/6ce2b94e15b97aa4da33f67962afddc2.png)
We call heapify starting at the first node that is the root of a tree of height at least $2$.
We can get this node by taking last child's node, so $\left[ \frac{\text{heapsize}}{2} \right]$.

- A node at height $h$ takes $h-1$ steps to fix.
- At height $h$, there are at most $\frac{n}{2^{n}}$ nodes.

$$
\begin{align}
&\sum^{[\log n] + 1}_{n=2}
(\text{num trees of height } h) \times (h-1) \\[6pt]
&= \sum^{[\log n] + 1}_{n=2}
\frac{n}{2^{h}} \times (h-1) \\[6pt]
&\leq n \times \sum^{\infty}_{n=2} \frac{h-1}{2^{h}}
\\[6pt]
&= n \times \text{constant}
\end{align}
$$

Therefore, building a heap takes $\Theta(n)$ time.

---
## Heaps and Sorting
Given an array, how can we use a heap to efficiently sort the array?

**Soln**:
We can convert the array into a [[heap]] ($\Theta(n)$ time) and repeatedly using [[#Extract-Max|extract-max]], updating and balancing the tree and decrementing the size.

In total, it takes $\Theta(n \log n)$ time.

### Example
Say we have this array $[5, 4, 9, 7]$ and we want to sort it in increasing order.

We can first turn the array into a heap.
$$
[5, 4, 9, 7] \to [9, 7, 5, 4]
$$
Then, we continuously do [[#extract-max]] until the array is sorted.
1. $[9, 7, 5, 4]$
2. Extract-max, update heap: $[7, 4, 5, 9]$.
3. Extract-max, update heap: $[5, 4, 7, 9]$.
4. Extract-max, update heap: $[4, 5, 7, 9]$.
   Done

---
## Max vs Min
So far, we've been working with [[Priority Queue|max priority queues]] and [[Heap|max heaps]]. There are min priority queues and min heaps too.

---
## See Also
- [[Priority Queue]]
- [[Graph]]
