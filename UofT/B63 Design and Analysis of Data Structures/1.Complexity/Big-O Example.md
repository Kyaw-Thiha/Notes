# Examples
1. Consider the insertion sort function.
   Prove that it is $\Theta(n^{2})$.

![image|400](https://notes-media.kthiha.com/Running-Time-of-Algorithms/e05058186c5c65d9c5f3f68a02ab1904.png)

**Solution**:
To prove that its $\Theta(n^{2})$, we need to prove that its both $O(n^{2})$ and $\Omega(n^{2})$.

**Part-1**: $O(n^{2})$
- Recall that [[Big-O]] provides an upper bound.
- Consider $\text{lines-}5-7$.
  Suppose that it loops $n$ times and it takes $x$ steps.
  Furthermore, suppose the loop test takes $y$ steps.
  Then, we have $n.x + y$.
- Let $z$ be the number of steps for $\text{lines-}1-4,8,9$ and the loop test.
- $\text{Lines-}2-9$ loop at most $n-1$ times.

$\therefore$ The function takes $n(nx + y + z)$ steps at most:
$$
n(nx + y + z)
= xn^{2} + n(y+z) \in O(n^{2})
$$
> Note that we **overcounted**/**overestimated** the number of times both the inner loop and outer loops loop.

**Part-2**: $\Omega(n^{2})$
- Recall that [[Big-Omega]] provides a lower bound.
- Consider the input that forces the greatest number of steps.
  It is an array of length $n$ that is sorted in decreasing order.

$$
[n-1, \ n-2, \ \dots, 2, \ 1, \ 0]
$$
- Suppose we ran the function using the list above and we will count $1$ per line.

![image|400](https://notes-media.kthiha.com/Big-O-Example/8d790676186d0b380d3d6d0d3c20529d.png)

Therefore, the function takes
$$
\sum ^{n-1}_{i=1} 3i+1 \text{ steps}
$$
Hence,
$$
\begin{align}
\sum ^{n-1}_{i = 1} 3i + 1
&= \frac{3}{2} n^{2} - \frac{n}{2} - 1 \\[6pt]
&\in \Omega(n^{2})
\end{align}
$$

Therefore, the function $\in \Theta(n^{2})$.

---
2. Prove $n^{3} - n^{2} + 5 \in \Theta(n^{3})$

**Solution**:
We have to prove $n^{3} - n^{2} + 5 \in O(n^{3})$ and $n^{3} - n^{2} + 5 \in \Omega(n^{3})$.

To prove $n^{3} - n^{2} + 5 \in O(n^{3})$:
$$
\begin{align}
n^{3} - n^{2} + 5  
&\leq n^{3} + 5 \\[6pt]
&\leq n^{3} + 5n^{3}
\end{align}
$$
When $n \geq 2$, $n^{3} + 5 n^{3} = 6n^{3}$.
Let $n_{0} = 1$ and $c=6$.
Then, $n^{3} - n^{2} + 5 \in O(n^{3})$.

To prove $n^{3} - n^{2} + 5 \in \Omega(n^{3})$:
$$
\begin{align}
n^{3} - n^{2} + 5 
&> n^{3} - n^{2} \\[6pt]
&\geq b.n^{3}
\end{align}
$$
Dividing both sides by $n^{2}$, we get
$$
\begin{align}
n - 1 &\geq b.n \\[6pt]
n - b.n &\geq 1 \\[6pt]
n &\geq \frac{1}{1-b}
\end{align}
$$
Since we want $n \geq n_{0}$, where $n_{0} \in \mathbb{N}$, we should pick $b < 1$.
Let $b = \frac{1}{2}$.
Then, $n=2$ and $n_{0}=2$.

---
## See Also
- [[Big-O]]
- [[Big-Omega]]
- [[Big-Theta]]
