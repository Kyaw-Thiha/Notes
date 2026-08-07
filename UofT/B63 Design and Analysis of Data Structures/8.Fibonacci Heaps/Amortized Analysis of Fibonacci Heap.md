# Amortized Cost
Recall that the [[Amortized Analysis|amortized time]] is 
$$
\text{actual cost} + \text{change in potential}
$$

- When we [[Fibonacci Heap|decrease_priority()]], we mark and/or move nodes to the root list and unmark.
- When we [[Fibonacci Heap|extract_min()]], we turn root nodes into child nodes.
- We can think of this as each time we mark a node, it will take $2$ steps to get it back into an unmarked child position.

This leads to the [[Potential Method (Amortized Analysis)|potential function]] of
$$
\phi(H) = \text{trees(H)} + 2 \cdot \text{marks(H)}
$$
where $\phi(H_{0}) = \text{tree}(H_{0}) + 2 \cdot \text{marks}(H_{0}) = 0$.

---
### Potential Function for `insert()`
When an [[Fibonacci Heap|insert()]] is performed, the [[Potential Method (Amortized Analysis)|potential changes]] by $1$.
$$
\begin{align}
\Delta(\phi)
&= \phi(H_{i+1}) - \phi(H_{i}) \\[6pt]
&= \text{tree}(H_{i+1}) + 2\cdot \text{mark}(H_{i+1})
- \text{tree}(H_{i}) - 2 \cdot \text{mark}(H_{i})  
\\[6pt]
&= 1
\end{align}
$$

---
### Potential Function for `decrease_priority()`
- Suppose we make $x$ cuts.
  For each cut made, we gain a root node (I.e. A tree)
- We are done cutting when we reach an unmarked node, possibly a root. 
- $x$ or $x-1$ nodes may have been marked.
  Furthermore, we may or may not mark the last node.
- $H_{i+1}$ will lose at least $x-1$ marks, but may gain $1$.

$$
\begin{align}
\text{marks}(H_{i+1})
&\leq \text{marks}(H_{i}) - (x-1) + 1 \\[6pt]
&= \text{marks}(H_{i}) - x + 2
\end{align}
$$
The amortized cost for `decrease_priority()` is 

$$
C_{i} + \phi(H_{i+1}) - \phi(H_{i})
$$

Computing it, we get
$$
\begin{align}
&\phi(H_{i+1}) - \phi(H_{i}) \\[6pt]
&= \text{tree}(H_{i+1})  
+ 2 \cdot \text{mark}(H_{i+1}) 
- \left( \text{tree}(H_{i}) + 2 \cdot \text{mark}(H_{i}) \right) \\[6pt]

&= \text{tree}(H_{i+1})  
+ 2 \cdot \text{mark}(H_{i+1}) 
- \text{tree}(H_{i}) - 2 \cdot \text{mark}(H_{i}) \\[6pt]

&= \text{tree}(H_{i+1}) - \text{tree}(H_{i})
+ 2 \ (\text{mark}(H_{i+1}) - \text{mark}(H_{i}) )  
\\[6pt]

&\leq x + 2(-x + 2) \\[6pt]

&= 4 - x
\end{align}
$$

Therefore, the amortized cost is
$$
\begin{align}
&(\text{\# cuts} + 1) + 4 - x \\[6pt]
&= (x-1) + 4 -x \\[6pt]
&= 5 \\[6pt]
&= O(1)
\end{align}
$$

---
### Potential Function for `extract_min()`
Recall that the actual cost is $O(\text{tree}(H) + d(n))$.

Each time we do `extract_min`, all of the marked children becomes unmarked.
$$
\therefore \text{mark}(H_{i+1}) \leq \text{mark}(H_{i})
$$
After an `extract_min` and a `consolidate`, we have $d(n) + 1$ roots. This is true if each root has an unique degree.

$\text{E.g:}$ Consider the [[Fibonacci Heap|fib heap]] below.
![image|300](https://notes-media.kthiha.com/Amortized-Cost-of-Fibonacci-Heap/a71cd6eac36ebf7cd5a9e4f807b9cee4.png)
Since there are $d(n) + 1$ roots after an `extract_min` and a `consolidate`, 
$$
\text{trees}(H_{i+1}) \leq d(n) + 1
$$

Since $\phi(H) = \text{tree}(H) + 2 \cdot \text{mark}(H)$,
$$
\Delta(\phi)
= \text{tree}(H_{i+1}) - \text{tree}(H_{i}) + \underbrace{2 \ (\text{mark}(H_{i+1}) 
- \text{mark}(H_{i}))}_{\text{This is less than } 0}
$$

$\therefore \Delta(\phi) \leq d(n) + 1 - \text{tree}(H_{i})$ and the amortized cost is $O(d(n))$.

---
### Bound on d(n)
We need to find a bound on $d(n)$. To do this, we need to determine the minimum number of nodes that is possible in a tree with a root of degree $k$.

We will denote this number as $N(k)$.
![image|300](https://notes-media.kthiha.com/Amortized-Cost-of-Fibonacci-Heap/b58ebd853c6e49df165750eda326be7a.png)

Recall the `golden ratio` $\phi$ which equals to $\frac{1 + \sqrt{ 5 }}{2} \approx 1.61803\dots$

For all integers $k \geq 0$, 
$$
F(k+2) \geq \phi^{k}
$$
Since $N(k) = F(k+2)$, 
$$
N(k) \geq \phi^{k}
$$
Let $n$ be the number of nodes in a tree with degrees $k$.
$$
\begin{align}
&n \geq N(k) \geq \phi^{k} \\[6pt]
&\log_{\phi}(n) \geq k
& \text{where k is d(n)}
\end{align}
$$

Therefore, `extract_min` has an [[Amortized Analysis|amortized cost]] of $O(\log n)$.

---
## Summary
- `insert`: Amortized cost $O(1)$
- `extract_min`: Amortized cost $O(\log n)$
- `decrease_priority`: Amortized cost $O(1)$

---
## See Also
- [[Fibonacci Heap]]
- [[Extract-Min Fibonacci Heap]]
- [[Decrease-Key Fibonacci Heap]]
- [[Amortized Analysis of Fibonacci Heap]]
- [[Amortized Analysis]]