# Decrease-Key
The pseudocode of `decrease_key(H,x,k)` is
![image|300](https://notes-media.kthiha.com/Decrease-Key-Fibonacci-Heap/47687863401bc982872b60c7d73c346c.png)
![image|300](https://notes-media.kthiha.com/Decrease-Key-Fibonacci-Heap/1a0669861c0864d1a0360035946a8242.png)
![image|300](https://notes-media.kthiha.com/Decrease-Key-Fibonacci-Heap/9812c39138b328b7f9c38e4ab2b3dcba.png)

---
### Explanation
- If $k$ is greater than $x$'s priority, we don't do anything cause we want to decrease $x$'s priority.
- Otherwise, we get $x$'s parent.
  If $x$'s parent is not NULL (ie if $x$ isn't on the root list), and $x$'s priority is less than the priority of its parent, we remove $x$ from its parent and insert it into the root list.
- After we cut $x$ from its parent, we do a `cascading_cut` on $x$'s parent. 
	- If $x$ is the first child to be cut, then we set $x$'s parent's `mark value` to be True.
	- If $x$ is second child to be cut, we also cut $x$'s parent.
	- We also do a `cascading_cut` on $x$'s grandparent.

---
### Example
Consider the [[Fibonacci Heap|fib heap]] below.
![image|300](https://notes-media.kthiha.com/Decrease-Key-Fibonacci-Heap/14f0826556f4cc218acc83eddaa1167d.png)

To decrease its priority, we do the following.
![image|300](https://notes-media.kthiha.com/Decrease-Key-Fibonacci-Heap/845cf1fa34e5d49659e8938016bd8010.png)
![image|300](https://notes-media.kthiha.com/Decrease-Key-Fibonacci-Heap/8ce05ad36298a3657037116b7256392a.png)
![image|300](https://notes-media.kthiha.com/Decrease-Key-Fibonacci-Heap/178370a8795556d88af63d97d1a5ecfb.png)
![image|300](https://notes-media.kthiha.com/Decrease-Key-Fibonacci-Heap/b9bb7c2f51751b905ae8296e0e326233.png)

---
## See Also
- [[Fibonacci Heap]]
- [[Extract-Min Fibonacci Heap]]
- [[Decrease-Key Fibonacci Heap]]
- [[Amortized Analysis of Fibonacci Heap]]