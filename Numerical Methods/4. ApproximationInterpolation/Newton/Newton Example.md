# Newton Example
#numerical-methods/interpolation/newton 

`Newton`
$$
P(x) = \sum^{n}_{i=0} \left[ a_{i} \ \prod^{i-1}_{j=0} (x-x_{j}) \right]
$$
where $a_{i}$ is the $i^{th}$ `divided difference` on $[x_{0}, \  x_{1}, \dots, \  x_{i}]$ 

---
`Question`

Compute the `quadratic polynomial` interpolating $\{ (0,3), \ (1,7), \ (2, 37) \}$ using the [[Newton Basis]].

---
`Solution`

$$
P(x) = a_{0} + (x-x_{0})a_{1} 
+ (x-x_{0})(x-x_{1})\ a_{2}
$$
where 
- $a_{0} = y[x_{0}] = y_{0}$
- $a_{1} = y[x_{1}, \ x_{0}] = \frac{y_{1} - y_{0}}{x_{1} - x_{0}}$

![[Newton Example.png|500]]

Hence, $P(x) = 3 + 4x + 13x\ (x-1)$. 

---
`Checking`

Checking at first point,
$$
\begin{align}
P(0)  
&= 3 \\[6pt]
&= y(0)
\end{align}
$$
Checking at second point,
$$
\begin{align}
P(1)
&= 3 + 4 \\[6pt]
&= 7 \\[6pt]
&= y(1)
\end{align}
$$
Checking at third point,
$$
\begin{align}
P(2)
&= 3 + 4(2) + 13(2)(1) \\[6pt]
&= 3 + 8 + 26 \\[6pt]
&= 37 \\[6pt]
&= y(2)
\end{align}
$$

Expanding $P(x)$ to [[Vandermonde Example|Monomial Basis]],
$$
\begin{align}
P(x)
&= 3 + 4x + 13x(x-1) \\[6pt]
&= 3 + 4x - 13x + 13x^2 \\[6pt]
&= 3 - 9x + 13x^2
\end{align}
$$

---
## See Also
- [[Newton Basis]]
- [[Vandermonde Example]]
- [[Lagrange Example]]