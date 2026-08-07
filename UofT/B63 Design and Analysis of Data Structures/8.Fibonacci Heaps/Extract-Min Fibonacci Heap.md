# Extract-Min
- The `min key` is already pointed to by $\text{H.min}$.
- We remove the min root, and promote its children to the root list.
- Now, $\text{H.min}$ may point to any node on the root list.
  (May not be the correct node).
- We consolidate the [[Fibonacci Heap|fib heap]]. 
  I.e: We want to end up with a root list with nodes of unique degrees.

---
## Pseudocode
- Repeat until all nodes in the `root list` have unique degrees.
	- Walk through the `root list`.
	- Remember the degree of each node passed. We can do this by using an array of pointers.
	- $A[1]$ has a pointer to the root node with degree $1$.
	  $A[2]$ has a pointer to the root node with degree $2$.
	  $\quad \vdots$
	- If we find a node $x$ with the same degree as an already seen node $y$ pointed to by $A$, we can `remove_max(x,y)` and make one the child of other.
- Find the new `min root` and set $\text{H.min}$ to it.

---
## Example
Consider the [[Fibonacci Heap]] below.
![image|300](https://notes-media.kthiha.com/Fibonacci-Heap/ccbe6cb261978e5bc11158bb2adf03ac.png)

Here are the steps if we do `extract_min` on the [[Fibonacci Heap|fib heap]].

![image|300](https://notes-media.kthiha.com/Extract-Min-Fibonacci-Heap/f2f459700bc78480cc9f8180a0dd9d55.png)
![image|300](https://notes-media.kthiha.com/Extract-Min-Fibonacci-Heap/44f51ce8a3e4f1226cab05dd05c9d5fd.png)
![image|300](https://notes-media.kthiha.com/Extract-Min-Fibonacci-Heap/f01d37959843e7cebe4b669ed7230e83.png)
![image|300](https://notes-media.kthiha.com/Extract-Min-Fibonacci-Heap/dbba986faf102aa1a98ab9b8b4601b0a.png)
![image|300](https://notes-media.kthiha.com/Extract-Min-Fibonacci-Heap/1024a8d0a009efabc043c4e06d8f5ace.png)

---
## See Also
- [[Fibonacci Heap]]
- [[Extract-Min Fibonacci Heap]]
- [[Decrease-Key Fibonacci Heap]]
- [[Amortized Analysis of Fibonacci Heap]]
