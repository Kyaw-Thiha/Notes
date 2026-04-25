# Existance of Interpolating Qudrature Rules

`Theorem`
Given any set of $n+1$ distinct nodes, we can choose the weights $A_{i}$ $s.t.$ 
- the quadrature rule is exact for all polynomials degrees $\leq n$
- and the weights are `unique`.

---
`Proof`

Since $Q(F)$ is a `linear operator`, we only need to show that $Q$ is exact for $1, \ x^1, \ x^2, \ \dots, x^n$.
For $x^k$,
$$
Q(x^k) = \sum^n_{i=0} A_{i} \ x_{i}^k
= \int^b_{a} x^k \ dx
=\frac{1}{k+1} (b^{k+1} - a^{k+1})
$$

Hence, we have $n+1$ equations.

Applying [[Vandermonde Theorem|Transposed Vandermonde]],
$$
\begin{bmatrix}
1 & 1 & \dots  & 1 \\
x_{0} & x_{1} & \dots & x_{n} \\
x_{0}^2 & x_{1}^2 & \dots & x_{n}^2 \\
\vdots & \vdots & & \vdots \\
x_{0}^k & x_{1}^k & \dots & x_{n}^k \\
\end{bmatrix}
\begin{bmatrix}
a_{0} \\
a_{1} \\
\vdots \\
a_{n}
\end{bmatrix}
= \begin{bmatrix}
b-a \\
\frac{b^2 - a^2}{2} \\
\vdots \\
\frac{b^{k+1} - a^{k+1}}{k+1}
\end{bmatrix}
$$

Hence, we can use the same argument as in [[Vandermonde Theorem]] to conclude that 
$$
\boxed{\ \text{The weights exists and are computable.} \ }
$$

---
## See Also 
- [[Interpolatory Quadrature]]