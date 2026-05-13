# Splay Tree
[[Splay tree]] brings most recently accessed or inserted elements to the root of the tree.

This allows search, insertion, and deletion operations in $O(\log n)$ [[Amortized Analysis|amortized time complexity]].

---
## Rotation Operations
- **Zig Rotation**:
	- if node has right child, perform right rotation.
	- if node has left child, perform left rotation.
- **Zig-Zag Rotation**: 
	- if the node has a right child and the right child has a left child, perform a right-left rotation.
	- if the node has a left child and the left child has a right child, perform a left-right rotation.

---
### Zig Rotation
![|300](https://media.geeksforgeeks.org/wp-content/uploads/20230203100633/Zig-rotation.png)

---
### Zag Rotation
![|300](https://media.geeksforgeeks.org/wp-content/uploads/20230203101229/zag-rotation.png)

---
### Zig-Zig Rotation
![|300](https://media.geeksforgeeks.org/wp-content/uploads/20230203102114/Zig-zig-rotation.png)

---
### Zag-Zag Rotation
![|300](https://media.geeksforgeeks.org/wp-content/uploads/20230203103016/zag--zag-rotation.png)

---
### Zig-Zag Rotation
![|300](https://media.geeksforgeeks.org/wp-content/uploads/20230203104532/Zig-zag-rotation2.png)

---
### Zag-Zig Rotation
![|300](https://media.geeksforgeeks.org/wp-content/uploads/20230203105833/zag-zig-rotation.png)

---
## Applications
Usually used where a ranking can be useful.
- Caching.
  Most frequently accessed items are moved to the top.
- File System
- Database indexing
- Graph algorithms
- Online gaming

### Benefits
- [[Amortized Analysis|Amortized time complexity]] of $O(\log n)$ for many operations.
- Self-adjusting

### Drawbacks
- [[Time Complexity|Worst time complexity]] of $O(n)$ for some operations.

---
