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
## See Also
- [[1-Norm]]
- [[Infinity Norm]]