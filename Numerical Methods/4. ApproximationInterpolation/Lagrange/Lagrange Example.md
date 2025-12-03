# Lagrange Matrix
#numerical-methods/interpolation/lagrange 


`Lagrange`
$$
P(x) = \sum^{n}_{i=0} l_{i}(x) \\ y_{i}
$$
where $l_{i}(x) = \Pi^n_{\substack{j=0 \\ j\neq i}} \frac{x - x_{j}}{x_{i} - x_{j}}$

---
`Question`

Compute the `quadratic polynomial` interpolating $\{ (0,3), \ (1,7), \ (2, 37) \}$ using the [[Lagrange Matrix]].

---
`Solution`

`Lagrange Multipliers`
$$
\begin{align}
l_{0}(x)
&= \frac{x-1}{0-1} \cdot \frac{x-2}{0-2} \\[6pt]
&= \frac{(x-1)(x-2)}{2}
\end{align}
$$

$$
\begin{align}
l_{1}(x)
&= \frac{x-0}{1-0} \cdot \frac{x-2}{1-2} \\[6pt]
&= -x(x-2)
\end{align}
$$

$$
\begin{align}
l_{2}(x)
&= \frac{x-0}{2-0} \cdot \frac{x-1}{2-1} \\[6pt]
&= \frac{x(x-1)}{2}
\end{align}
$$

`Forming the Polynomial`
$$
p(x) 
= \frac{3(x-1)(x-2)}{2} - 7x(x-2) 
+ \frac{37x(x-1)}{2}
$$

---
`Checking`
$$
\begin{align}
P(0)
&= \frac{3(-1)(-2)}{2} \\[6pt]
&= 3 \\[6pt]
&= y(0)
\end{align}
$$
$$
\begin{align}
P(1)
&= -7(-1) \\[6pt]
&= 7 \\[6pt]
&= y(1)
\end{align}
$$

$$
\begin{align}
P(2)
&= \frac{37(2)(2-1)}{2} \\[6pt]
&= 37 \\[6pt]
&= y(2)
\end{align}
$$

---
`Forming the Monomial Basis`

Let's expand and simplify $P(x)$.
$$
\begin{align}
P(x)
&= \frac{3(x-1)(x-2)}{2} - 7x(x-2) +  
\frac{37x(x-1)}{2} \\[6pt]
&= \frac{3}{2} (x^2 - 3x +2)
-7 (x^2 - 2x) + \frac{37}{2} (x^2 - x) \\[6pt]
&= \frac{3x^2}{2} - \frac{9x}{2} + 3
-7x^2 + 14x + \frac{37x^2}{2} - \frac{37x}{2} \\[6pt]
&= 3 - 9x + 13x^2
\end{align}
$$

Note that $3 - 9x + 13x^2$ is the same as the [[Vandermonde Example|Monomial Basis]].

---
## See Also
- [[Lagrange Matrix]]
- [[Vandermonde Theorem]]
- [[Vandermonde Example]]
