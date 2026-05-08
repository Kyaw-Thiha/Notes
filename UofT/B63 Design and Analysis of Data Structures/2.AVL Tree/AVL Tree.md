# AVL Tree

An [[AVL Tree]] is a self-balancing BST where the difference between the heights of the left and right subtrees cannot be more than $1$ for all nodes.

![AVL Tree|300](https://learnersbucket.com/wp-content/uploads/2021/03/AVL-Tree-in-Javascript.png)

---
## Properties

- In an [[AVL Tree]], the value of the left subtree is less than the value of the parent node.
  Furthermore, the value of the right subtree is greater than the value of the parent node.
- The height of an [[AVL Tree]] is $O(\log n)$
- Each internal node has balance property equal to $-1, \ 0, \ 1$
  It ensures that the height is always a function $\log(n)$
- $\text{Balance Value} = \text{height of left sub-tree} - \text{height of right sub-tree}$

---
## Examples

![image|400](https://notes-media.kthiha.com/AVL-Tree/a93592b0740d1b9b1a2a5823d1eaf768.png)

---
## Operations
There are $3$ operations: insert, delete and search.
In BST, the worst case complexity for $3$ operations is $O(n)$.
In [[AVL Tree]], it is $O(\log n)$.

**Searching**: 
Search in an [[AVL Tree]] is the same as a BST.

**Inserting**: 
Consider the following AVL Tree.

![image|200](https://notes-media.kthiha.com/AVL-Tree/61599078b0eb2c323c8ef80770f61c3e.png)

For balancing the trees, different [[Rotations in AVL Tree|rotations in AVL Tree]] are used as required.

---
$\text{(a)}$ Update the tree and the balance values of each node, after inserting $6$.

Since $6 < 44$, we go to the left subtree.
Since $6 < 17$, we insert $6$ as $17$'s left child.
Furthermore, $17$ now has a left subtree and a right subtree of the same height, so $17$'s balance value is now $0$.

---
$\text{(b)}$ Update the tree and the balance values of each node, after inserting $35$.
![image|200](https://notes-media.kthiha.com/AVL-Tree/82b4d7e0ffcb466f9b039d0ffb9ddb0e.png)

Note that the left subtree is no longer balanced.
$17$ now has a balance value of $-2$.

We can fix the problem with a **single rotation**.
If we rotate ccw, $32$ moves up and $17$ comes down.
![image|200](https://notes-media.kthiha.com/AVL-Tree/0bbdd96c2189a99d39571bc8fa844b94.png)

---
$\text{(c)}$ Update the tree and the balance values of each node, after inserting $45$.
![image|200](https://notes-media.kthiha.com/AVL-Tree/bee5df7566395fa909adc3930c0f0357.png)

The tree is out of balance and to fix it, we need to do **single rotation**.
If we rotate clockwise about $78$, $50$ goes up and $78$ comes down.
Furthermore, $62$ becomes the left child of $78$.
![image|200](https://notes-media.kthiha.com/AVL-Tree/7838de87ec4482e70f25301bc55eb194.png)

---
$\text{(d)}$ Update the tree and the balance values of each node, after inserting $46$.
![image|200](https://notes-media.kthiha.com/AVL-Tree/00c6cc9139691da7f91a82cd8726b18d.png)

This time, we need a **double rotation** to balance the tree.

First we rotate counter-clockwise about $45$.
The tree becomes 
![image|200](https://notes-media.kthiha.com/AVL-Tree/27238233b6024d6e0cab5509dc9af6ee.png)

Then, we rotate clockwise about $48$.
![image|200](https://notes-media.kthiha.com/AVL-Tree/2264c28c34bd2a990b49a3a00bc98242.png)

We know that we need a double rotation because there was a change in the sign of the balance values.

Note that after $46$ was inserted, we see
![image|200](https://notes-media.kthiha.com/AVL-Tree/d5cec6f31759602fcdd76dec9b5dcdb1.png)

If there is a $2$ followed by $-1$ or a $-2$ followed by a $1$, we need to do a double rotation.

---
**Deleting**: Delete by finding the successor.
Go down the left-most path of the node's right subtree to get the smallest node in that subtree.
Replace the deleted node with that node.

Then, rebalance the tree as needed.

---
## AVL Tree Algorithms
The basic building blocks of [[AVL Tree Algorithms]] are splitting and merging.

They can be used to implement
- [[Union of AVL Trees]]
- [[Intersection of AVL Trees]]
- [[Difference of AVL Trees]]

---
## See Also
- [[AVL Tree Height]]
- [[Rotations in AVL Tree]]
- [[AVL Tree Algorithms]]
- [[Union of AVL Trees]]
- [[Intersection of AVL Trees]]
- [[Difference of AVL Trees]]

