# Error in Interpolatory Quadrature

The error in [[Interpolatory Quadrature]] can be computed as
$$
\begin{align}
l_{n}
&= I(F) - Q_{n}(F) \\[6pt]
&= \int^b_{a} F(x) - P_{n}(x) \ dx \\[6pt]
&= \int^b_{a} \frac{F^{(n+1)}(\epsilon)}{(n+1)!}
\prod^n_{i=0} (x-x_{i}) \ dx
\end{align}
$$

Note that we applied [[Error in Polynomial]].

---
`Theorem: Convergence Guarantee`

`Let` $F \in C[a,b]$ and $Q_{n}(F) = \sum^n_{i=0} A_{i}^{(n)} F(x_{i}^{(n)})$.
`Then`, 
$$
\begin{align}
&\lim_{ n \to \infty } Q(F) = I(F) \\[6pt]
\iff & \exists \text{ constant } l_{e} \ s.t. \ 
\sum^N_{i=0} |A_{i}^{(n)}| \leq l_{e}, \ \forall n
\end{align}
$$

In other words, 
$$
\boxed{\ \text{Interpolatory Quadrature Rules converge to $I(F)$ as degree $n\to \infty$} \ }
$$

---
## See Also
- [[Interpolatory Quadrature]]
