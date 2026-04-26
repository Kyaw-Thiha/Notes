# Big Omega
Let $g \in \mathcal{F}$. 
Then, $\Omega(g)$ is the set of functions $f \in \mathcal{F}$ $s.t.$ 
$\exists b \in \mathbb{R}^{+}, \ \exists n_{0} \in \mathbb{N}, \ \forall n \in \mathbb{N}, \ n\geq n_{0}$
$$
f(n) \geq b. g(n) \geq 0
$$

![Big-Omega|400](https://media.geeksforgeeks.org/wp-content/uploads/20240329124349/big-omega-image.webp)
This shows $f(n) \in \Omega(g(n))$.

[[Big-Omega]] provides a lower bound.
If an algorithm is $\Omega(x)$, then the algorithm takes at least $c.x$ steps to run on the worst possible input.

To show if an algorithm is $\Omega(x)$, we have to show that there is an input that makes the algorithm takes at least $c.x$ steps.

---
## See Also
- [[Big-O]]
- [[Big-Theta]]
- [[Running Time of Algorithms]]
