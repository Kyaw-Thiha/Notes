# p-Norm

$L_{p}\text{-Norm}$ are different ways of measuring the length of a vector, defined as
$$
||x||_{p}
= \left( \sum^d_{i=1} |x_{i}|^{p} \right)^{1/p}
$$
where $p \in [0, \ \infty[ \ \cup \ \{ \infty \}$.

![p-Norm Visualization|300](https://miro.medium.com/v2/1*_Jo97QAjgKl2W9mJeI6Lxg.png)

---
## Special Cases
- $L_{0}$: $||\mathbf{x}||_{0} = \sum^d_{i=1} \mathbb{1}(x_{i} \neq 0)$ (Note that this is not really a norm)
- $L_{1}$: $||\mathbf{x}||_{1} = \sum^d_{i=1} |x_{i}|$
  [[1-Norm|Read more]]
- $L_{2}$: $||\mathbf{x}||_{2} = \sqrt{ \sum^d_{i=1} x_{i}^2 }$ 
- $L_{\infty}$: $||\mathbf{x}||_{\infty} = \max \{ |x_{1}|, \ |x_{2}|, \ \dots, \ |x_{d}| \}$
  [[Infinity Norm|Read more]]

---
## Probability Distribution
Let $v \in \mathcal{M(S)}$ be the probability distribution.
Then for a function $V \in \mathcal{F}$, we define $L_{p}(v)$-norm of $V$ with $1\leq p < \infty$ as
$$
||V||^{p}_{p,v}
\triangleq \int_{\mathcal{S}} |V(s)|^{p}
\ dv(s)
$$
The $L_{\infty}(\mathcal{S})$-norm is defined as
$$
||V||_{\infty} \triangleq \sup_{s \in \mathcal{S}} |V(s)|
$$

---
## See Also
- [[1-Norm]]
- [[Infinity Norm]]