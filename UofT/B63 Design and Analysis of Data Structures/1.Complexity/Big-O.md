# Big-O Notation

- Let $\mathbb{R}^{+}$ be the set of positive, real numbers.
- Let $\mathbb{R}^{+}_{0}$ be the set of positive, real numbers $\geq 0$.
- Let $\mathbb{N}_{k}$ be the set of natural numbers $\geq k$.
- Let $\mathcal{F}$ be the set of functions of $f: N_{k} \to R^{+}_{0}$
- Let $g \in \mathcal{F}$.

Then, $O(g)$ is the set of functions $f \in \mathcal{F}$ $s.t.$ 
$\exists c \in \mathbb{R}^{+} ,\ \exists N_{0} \in \mathbb{N} , \ \forall n \in \mathbb{N}, \ n \geq N_{0}$
$$
0 \leq f(n) \leq c.g(n)
$$

![Big-0 Notation|400](https://media.geeksforgeeks.org/wp-content/uploads/20240329121436/big-o-analysis-banner.webp)
This shows $f(n) \in O(g(n))$. 

[[Big-O]] provides an upper bound.
If an algorithm is $O(x)$, then the algorithm takes at most $c.x$ steps to run on the worst possible input.

To show if an algorithm is $O(x)$, we have to show that for every input, the algorithm takes at most $c.x$ steps.
We can do this by overestimating the steps it takes.

---
## See Also
- [[Big-Omega]]
- [[Big-Theta]]
- [[Time Complexity]]
