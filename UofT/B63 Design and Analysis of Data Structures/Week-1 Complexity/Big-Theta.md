# Big-Theta 
**Informal Definition**: If $f(n) \in O(g(n))$ and $f(n) \in \Omega(g(n))$, 
then $f(n) \in \Theta(g(n))$.

**Formal Definition**:
Let $g \in \mathcal{F}$.
Then, $\Theta(g)$ is the set of functions $f \in \mathcal{F}$ $s.t.$ $\exists b \in \mathbb{R}^{+}, \ \exists c \in \mathbb{R}^{+}, \exists n_{0} \in \mathbb{N}, \ \forall n \geq n_{0}$,
$$
b.g(n) \leq f(n) \leq c.g(n)
$$

**Theorem**: $f(n) \in \Theta(g(n))$ $\iff$ $f(n) \in O(g(n))$ and $g(n) \in O(f(n))$.

![Big Theta|400](https://i.ytimg.com/vi/-FORaw3VgQE/maxresdefault.jpg)

The above picture shows $f(n) = \Theta(g(n))$.
[[Big-Theta]] provides a tight bound.

---
## See Also
- [[Big-O]]
- [[Big-Omega]]
- [[Running Time of Algorithms]]