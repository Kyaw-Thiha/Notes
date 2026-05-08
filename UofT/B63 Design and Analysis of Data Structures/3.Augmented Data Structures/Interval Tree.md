# Interval Tree
[[Interval Tree]] tree represents a closed time interval: 
$$
\{ x \in \mathbb{R} \mid l \leq x \leq h \} = [l,h]
$$
![image|300](https://notes-media.kthiha.com/Interval-Tree/a621903bc77b418a09668231b86dc0c9.png)

---
## Operations
1. `insert(l, h)`: Store `[l, h]` in the collection.
2. `delete(l, h)`: Delete `[l, h]`.
3. `search(l, h)`: Return a stored interval that overlaps with `[l, h]`.

> **Note-1**
> Consider an interval $[1, 5]$.
> Then, $[-1, 1], \ [-1, 5], \ [0, 5], \ [1, 6], \ [5, 8], \ [0, 10]$, etc overlaps it.
> However, $[-1, 0], \ [6, 10], \ [8, 13]$ does not overlap with $[1, 5]$.


> **Note-2**
> To compare $2$ intervals $[a,b]$ and $[l,h]$:
> 1. If $a < l$, then $[a, b] < [l, h]$.
> 2. If $a = l$ and $b < h$, then $[a,b] < [l,h]$.

Eg: $[1,3] < [2, 5]$ and $[2, 3] < [2, 5]$.

---
![image|300](https://notes-media.kthiha.com/Interval-Tree/a621903bc77b418a09668231b86dc0c9.png)

**Eg-1**:
Suppose we want to search $(10, 12)$.

Consider $[8,9]$.
It doesn't overlap with $[10, 12]$ and $[10, 12]$ lies to the right of $[8,9]$.

If $[10, 12]$ were to overlap with an interval in the left subtree of $[8,9]$, there must be some $h_{i} \geq 10$.
If there is an $h_{i} \geq 10$ in the left subtree, then it is guaranteed that it overlaps with $[10, 12]$.
We see that $[6, 10]$ overlaps with $[10, 12]$ in the left subtree.

**Eg-2**:
Now suppose we want to do $\text{search}(2, 4)$.
$[2,4]$ does not overlap with $[8,9]$ and is to the left of $[8,9]$.
Furthermore, $[2, 4]$ cannot overlap with any intervals in the right subtree of $[8,9]$ because their lower end is greater than or equal to $8$.

If $[2,4]$ were to overlap with an interval in the left subtree of $[8,9]$, there must be some $h_{i} \geq 2$.
We can see that $[0, 3]$ overlaps with $[2,4]$.

> Note that in both cases, there needed to be an $h_{i} \geq l$ in order for there to exists an overlapping interval in the left subtree.

---
## Augmenting the tree
We can [[Augmented AVL Tree|augment the tree]] by adding the $\max$ $h_{i}$ in the subtree rooted at $x$ to each node $x$.

![image|300](https://notes-media.kthiha.com/Interval-Tree/928ccbcbba10312aef99e05aebfa121e.png)

---
## Pseudocode for Search
![image|300](https://notes-media.kthiha.com/Interval-Tree/ba8549b3f9cfd5620bc75953b9cad66d.png)

---
## See Also
- [[Augmented AVL Tree]]
- [[Interval Tree]]
- [[Weight-Balanced Trees]]
- [[Weight-Balanced Tree Rotations]]
- [[Augmented Data Structures]]
