# Interpolatory Quadrature
#numerical-methods/interpolatory-quadrature

`Composite Rule` is to apply simple functions to many short intervals, and then sum them.

![Interpolatory Quadrature](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhCsB4sB_f4LfNEdG83SQggvGVPOi9WoQ8B3uhUoUhNdDc_vXQR5vkzg1u21N90t1R2qPXaGC0XSspbZXd7p7fNNIxDFOYuZFD14iqcbq9c4sX_oMifczZgjuy_ZAcLHzX4EIXwotfX_fU/s1600/trapezoid_2_40.gif)

---
`Basic Formula`
$$
\begin{align}
\underbrace{I(F)}_{\text{exact}} &\approx Q(F)  
= \underbrace{I(p)}_{\text{exact}} \\[6pt]
I(F) &= \int^b_{a} p(x) dx
\end{align}
$$

`Lagrange Form`
In [[Lagrange Matrix|Lagrange Form]], $p(x) = \sum^N_{i=0} F(x_{i}) \ l_{i}(x)$ $\in P_{n}$
Hence,
$$
\begin{align}
Q(F)  
&= I(p) \\[6pt]
&= \int^b_{a} p(x) dx \\[6pt]
&= \int^b_{a} \sum^N_{i=0} F(x_{i}) p(x) \ dx \\[6pt]
&= \sum^N_{i=0} A_{i} F(x_{i})
\ , & A_{i} = \int^b_{a} p(x) dx
\end{align}
$$

---
`Standard Form of Qudrature Rule`

$$
Q(F) = \sum^N_{i=0} A_{i} F(x_{i})
$$
where 
- $A_{i}$ are `weights`
- $x_{i}$ are `nodes`

`Notes`
- $Q(F)$ is a `linear operator`: $Q(\alpha. F + G) = \alpha Q(F) + Q(G)$
- `If` $Q(F)$ integrates $1, \ x, \ x^2, \dots, \ x^n$ exactly
  `Then` $Q$ integrates all polynomials degrees $\leq n$ exactly.

---
`Theorem: Existance of Interpolatory Quadrature Rules`
Given any set of $n+1$ distinct nodes, we can choose the weights $A_{i}$ $s.t.$ 
- the quadrature rule is exact for all polynomials degrees $\leq n$
- and the weights are `unique`.

[[Existance of Interpolating Quadratures Rule|Read Proof Here]]

---
`Precision`

`If` $m$ is the largest natural numbers $s.t.$ $Q$ integrates all poly of degree $\leq m$ exactly
`Then` $m$ is the precision of $Q$

`Note`: $m$ may be greater than $n$.

---
## Interpolatory Quadrature Rules
- [[Midpoint Rule]]: Interpolates with the midpoint
  Has precision $m=1$
- [[Trapezoidal Rule]]: Interpolates with line between interval
  Has precision $m=2$
- [[Simpson's Rule]]: Interpolates with quadratic polynomial between interval
  Has precision $m=3$

---
`Error in Quadrature Rules`
Error can be computed as
$$
\begin{align}
l_{n}
&= \int^b_{a} \frac{F^{(n+1)}(\epsilon)}{(n+1)!}
\prod^n_{i=0} (x-x_{i}) \ dx
\end{align}
$$

We also have a `theorem` that guarantees that `quadrature rules` converge to $I(F)$ as degree $n \to \infty$.

[[Error in Interpolatory Quadrature|Read More]]

---
## See Also
- [Good blog post about Interpolatory Quadrature](https://g-biomech.blogspot.com/2014/08/numerical-integration-simpsons-rule_26.html)
- [[Existance of Interpolating Quadratures Rule]]
- [[Midpoint Rule]]
- [[Trapezoidal Rule]]
- [[Simpson's Rule]]
- [[Error in Interpolatory Quadrature]]