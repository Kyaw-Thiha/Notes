# B-Tree
[[B-Tree]] is a self-balancing $m$-way tree designed to optimize data access, especially on disk-storage systems.

![B-Tree|300](https://media.geeksforgeeks.org/wp-content/uploads/20240417151229/b-tree-in-python-banner.webp)

---
## Properties
![B Tree|300](https://media.geeksforgeeks.org/wp-content/uploads/20200506235136/output253.png)

- All leaf nodes of a [[B-Tree]] are at same level.
- The keys of each node of [[B-Tree]] at same depth should be stored in ascending order.
- All non-leaf nodes except root should have $\frac{m}{2}$ children.
- All nodes except root should have at least $\frac{m}{2}-1$ keys.
- If root is only node in [[B-Tree|tree]], then it have no children.
  Else, it has at least $2$ children and at least $1$ key.
- A non-leaf node with $n-1$ key values should have $n$ non-null children.

---
## Operations

- `search()`: $O(\log n)$
- `insert()`: $O(\log n)$ 
- `delete()`: $O(\log n)$
- `transverse()`: $O(n)$

### Search
![|300](https://media.geeksforgeeks.org/wp-content/uploads/20250114112546349145/btree_2.webp)
![|300](https://media.geeksforgeeks.org/wp-content/uploads/20250114112546512171/btree_3.webp)
![|300](https://media.geeksforgeeks.org/wp-content/uploads/20250114112546656532/btree_4.webp)
![|300](https://media.geeksforgeeks.org/wp-content/uploads/20250114112546818741/btree_5.webp)

---
## Application
- Large databases
- Servers
- CAD systems for geometric data

### Benefits
- [[Time complexity]] of $O(\log n)$ for most operations.
- Self-balancing
- Efficient storage utilization

### Drawbacks
- Has high disk usage.
- For small datasets, might be slower than BST.

---
## See Also
- [Good Explanation by GeeksForGeeks](https://www.geeksforgeeks.org/dsa/introduction-of-b-tree-2/)
- [[AVL Tree]]
- [[Augmented AVL Tree]]
- [[Interval Tree]]
