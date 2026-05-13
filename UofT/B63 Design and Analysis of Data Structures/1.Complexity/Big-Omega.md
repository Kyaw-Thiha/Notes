# Big Omega
Let $g \in \mathcal{F}$. 
Then, $\Omega(g)$ is the set of functions $f \in \mathcal{F}$ $s.t.$ 
$$
\exists b \in \mathbb{R}^{+}, \ \exists n_{0} \in \mathbb{N}, \ \forall n \in \mathbb{N}, \ n\geq n_{0}
\implies
f(n) \geq b. g(n) \geq 0
$$
Equivalently, $f \in \Omega(g) \iff g \in O(f)$.

![Big-Omega|400](https://media.geeksforgeeks.org/wp-content/uploads/20240329124349/big-omega-image.webp)
This shows $f(n) \in \Omega(g(n))$.

[[Big-Omega]] provides a lower bound.
If an algorithm is $\Omega(x)$, then the algorithm takes at least $c.x$ steps to run on the worst possible input.

To show if an algorithm is $\Omega(x)$, we have to show that there is an input that makes the algorithm takes at least $c.x$ steps.

---
## Equivalence to [[Big-O]]
**WTP**: $f \in \Omega(g(n)) \iff g(n) \in O(f(n))$.
Let $f \in \Omega(g(n))$.
Then, 
$$
\begin{align}
\exists n_{0} \in N, \forall n \in N, b \in \mathbb{R}  
\implies &f(n) \geq b.g(n) \\[6pt]
\implies &g(n) \leq \frac{1}{b} f(n) \\[6pt]
\implies &g(n) \in O(f(n))
\end{align}
$$

---
## Proof Example
Prove that $2n^{3} - 7n + 1 = \Omega(n^{3})$

We need to first find $n$ bound since it can be negative.
$$
\begin{align}
2n^{3} - 7n + 1
&= n^{3} + n^{3} - 7n + 1 \\[6pt]
&= n^{3} + n(n^{2} -7) \\[6pt]
&\implies n \geq 3
\end{align}
$$
Hence, we next find $c$
$$
\begin{align}
n^{3} - 7n + 1 &\geq 0 \\[6pt]
n^{3} + n^{3} - 7n + 1 &\geq 1 n^{3} \\[6pt]
\implies &c=1
\end{align}
$$

Choosing $n=3,\ c=1$, we get that
$$
\begin{align}
2n^{3} - 7n + 1
&= n^{3} + (n^{3} - 7n) + 1 \\[6pt]
&\geq n^{3} + 1 & \text{since } n\geq3 \\[6pt]
&\geq n^{3} \\[6pt]
&= cn^{3} &\text{with } c=1
\end{align}
$$
By definition of [[Big-Omega]], $2n^{3} - 7n + 1$ is in $\Omega(n^{3})$.

---
## See Also
- [[Big-O]]
- [[Big-Theta]]
- [[Time Complexity]]
