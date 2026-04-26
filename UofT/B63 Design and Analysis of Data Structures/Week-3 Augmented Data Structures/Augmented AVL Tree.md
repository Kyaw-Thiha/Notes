# Augmented AVL Tree
[[AVL Tree|AVL trees]] can be [[Augmented Data Structures|augmented]] by adding $2$ operations: 
- **rank(k)**: Given a key $k$, this returns its rank.
  (It's position among the elements in a data structure).
- **select(r)**: Given a rank $r$, this returns the key that has this rank.

---
### Example
![image|300](https://notes-media.kthiha.com/Augmented-AVL/fbcd8fb0e6c8cd4d119741ab3e2fb225.png)

---
### Implementation Ways

**Method-1**: Use [[AVL Tree]] without modification

- To find rank of a node, do a `in-order transversal` of tree.
  Keep track of the number of nodes visited until the desired node is reached.
- To find select of a rank, do `in-order transversal` of tree.
  Keep track of the number of nodes visited until the desired node is reached.

Even though search, insert and delete won't be affected, the worst [[Running Time of Algorithms|time complexity]] for rank and search is $\Theta(n)$.
This is because we may have to visit every node.

---
**Method-2**: Augment each node with with an additional field $\text{rank[x]}$ that stores $x$'s rank in the tree.

- Both rank and search will take $\Theta(\log n)$.
- However, insert and delete will take $\Theta(n)$.

This is because each time we add and delete a node, we have to update the rank field for all the nodes that come after it in an `in-order transversal`.

> Here, rank and select are efficient but not insert and delete.

---
**Method-3**: Augment each node with an additional field $\text{size[x]}$ that stores the number of  keys in the subtree rooted at $x$ including $x$.

![image|300](https://notes-media.kthiha.com/Augmented-AVL/fbcd8fb0e6c8cd4d119741ab3e2fb225.png)

Recall that $\text{Rank}(x) = 1 + \text{no. of keys after } x$.
Let $T_{L}$ be the left-child of $x$.

The relative $\text{Rank}(x)$ is equal to $\text{size}(T_{L}) + 1$.
This means that the rank of a node is related to the size of the subtree rooted at neighbouring nodes.

**Finding rank given key**
Given a key $k$ to find the rank of $k$, we search for $k$, keeping track of the rank of the current node.

Whenever we go down a right path, we add the size of the left subtree that we skipped and the key itself that we skipped.
When we find the key, to get its true rank, 
$$
\text{true rank} = 
\text{current rank} + \text{size of left child} + 1
$$

**Finding key given rank**
- Let $r$ be the rank of the key to be found.
- Let $x$ be $\text{root}(T)$.
  Start at the root and work down.
- Let $S$ be the left child.
  Compare the given rank $r$ to $\text{size}[S] + 1$.
- If they are equal, return $x$.
- If $(r < \text{size}[S] + 1)$, we know that the element we are looking for is in $S$, so we do the recursive call on $S$.
- If $(r > \text{size}[S] + 1)$, we know that the element we are looking for is in the right subtree.
  Therefore, the relative rank in the remaining elements ignoring $S$ is equal to $r - (\text{size}[S] + 1)$.
  We change $r$ accordingly and go down the right subtree of $S$.

---
**Complexity of using size field**
The complexity for rank is the same as search: $O(\log n)$.
For insert and delete, there are $2$ parts: operation then rebalancing.

**Operations**:
1. `insert(x)`: For each node visited when finding the position for $x$, increment its size.
2. `delete(x)`: 
   If $x$ is a leaf, then we transverse the path from $x$ to the root and decrement the size of each nodes on the path.
   If $x$ is not a leaf, we replace it with its successor $y$.
   Then, we transverse the path from $y$ to the root, decrementing each node along the path.

**Rotations**:
Each rotation, we only need to consider a constant number of nodes so it takes $\Theta(1)$ time.

$\therefore$ Each operation takes $\Theta(\log n)$ time in the worst case.

---
