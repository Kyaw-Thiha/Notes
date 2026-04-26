# AVL Tree Algorithms
For each algorithm $A(T, V)$ where $T$ and $V$ are $2$ [[AVL Tree|AVL Trees]], 
- assume $T$ to be taller
- and let $k$ be the root of $V$.

Then, 
- **Split** $T$ into $T < k$ and $T > k$.
  Let $V_{L}$ and $V_{R}$ be the subtrees rooted at $k$'s left and right children.
- **Compute** $L \leftarrow A(T<k, \ L)$ and $R \leftarrow A(T > k, \ R)$.
  This is the divide part.
- **Merge** $L$ and $R$ back together.
  Depending on if its union, intersection, or difference, we may also merge it with $k$ to get the required [[AVL Tree]].
  This is the conqueror part.

---
## Merging
Assume that there are $2$ [[AVL Tree]] $T$ and $V$ and assume that keys in $T$ are less than keys in $V$.

Let $k$ be a value that is greater than or equal to $T$'s largest key and smaller than $V$'s root.

There are $3$ cases to consider:
- $h(T) > h(V) + 1$
- $h(V) > h(T) + 1$
- $|h(V) - h(T)| \leq 1$

**Case-1**: $h(T) > h(V) + 1$
Go down $T$'s right-most path until we reach height of $h(V) + 2$.
Then, insert $k$ as that node's right child as that node's right child.

The previous old right subtree of that node is now $k$'s left child and $V$ is $k$'s right child.
Finally, we rebalance if needed.

**Example**
![image|300](https://notes-media.kthiha.com/AVL-Trees-Algorithms/7e09007afd87fa2ed307072dd52df08d.png)

**Solution**:
In this case, $V$ is the tree
![image|100](https://notes-media.kthiha.com/AVL-Trees-Algorithms/f94fe6f632730a7dd5308f678a82a761.png)
and it has the height of $2$.

This means that we have to go down the other tree's right-most path until we reach a height of $4$.
Then, we'll insert $k$ as its right child.

Let $k=15$.
![image|300](https://notes-media.kthiha.com/AVL-Trees-Algorithms/e1a89c2c9a90cf0acb6e07d972f9eaa3.png)

After you insert $k$, the right subtree of $5$ becomes the left subtree of $k$.
![image|300](https://notes-media.kthiha.com/AVL-Trees-Algorithms/f6b24da8f166d2abbf795fd2f72240f4.png)

Lastly, rebalance the tree as needed.
![image|300](https://notes-media.kthiha.com/AVL-Trees-Algorithms/70801f1000d16ee9832e74626260db19.png)

---
**Case-2**: $h(V) > h(T) + 1$
This is very similar to the first case.
Start at $V$'s root and go down its leftmost path until you reach a height of $h(T) + 2$.

Then, insert $k$ as that node's left child.
$T$ becomes $k$'s left child and old left subtree of the node becomes $k$'s right subtree.

Finally, rebalance.

---
**Case-3**: $| \ h(V) - h(T) \ | \leq 1$
The heights of $T$ and $k$ are within $1$.
Let $k$ be the root of a new [[AVL Tree]].
$T$ becomes $k$'s left subtree and $V$ becomes $k$'s right subtree.

---
## Splitting 
- Let $T$ be an [[AVL Tree]].
- Let $k$ be the value which we split $T$ by.
- Let $L$ be the $r$'s left child.
- Let $R$ be the $r$'s right child.
- Let $r$ be $T$'s root.
- Let $b$ be a boolean value, whose value depends on if $k$ is in $T$.

Consider this pseudo-code for splitting an [[AVL Tree]]:
![image|350](https://notes-media.kthiha.com/AVL-Trees-Algorithms/be672ef22a87b75fb6417abd3c30dadb.png)
![image|350](https://notes-media.kthiha.com/AVL-Trees-Algorithms/202142c84f909e7b74f80beabf0e8ddf.png)

---
## See Also
- [[AVL Tree]]
- [[AVL Tree Height]]
- [[Rotations in AVL Tree]]
- [[Union of AVL Trees]]
- [[Intersection of AVL Trees]]
- [[Difference of AVL Trees]]