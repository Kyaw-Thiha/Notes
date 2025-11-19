# Newton's Method
#numerical-methods/non-linear/methods/newton 

`Formula`
$$
x_{k+1} = x_{k} - \frac{F(x)}{F'(x)}
$$
This is the [[Fixed Point Methods (FPM)#Second Form|Second Form]] with $h(x) = \frac{1}{F'(x)}$

![Newton's Method](https://i.ytimg.com/vi/jXI6Sd6JIug/maxresdefault.jpg)

---
`Proof`
Suppose that $F(\tilde{x}) = 0$ and $F'(\tilde{x}) \neq 0$
$$
\begin{align}
g'(x) &= 1 - \left( \frac{F'(x) . F'(x) - F(x) . F''(x) } {(F'(x))^2} \right) \\[6pt]

&= \frac{(F'(x))^2 - (F'(x))^2 + F(x)\ F''(x) } {(F'(x))^2} \\[6pt]

&= \frac{F(x)\ F''(x) } {(F'(x))^2} \\[6pt]
&= 0
\end{align}
$$
By the [[Rate of Convergence Theorem]], `Newton's Method`  has at least quadratic convergence for any function $F$.

---
`Geometric Interpretation`
We want to solve $F(x) = 0$ at an initial guess $X_{k}$ 
Which is an approximate model to $F(x)$ by a linear polynomial $p(x)$ that satisfies the conditions

1. $p(x_{k}) = F(x_{k})$
2. $p^{'}(x_{k}) = F^{'}(x_{k})$

where $p(x)$ is the tangent line to $F(x)$

$$
p_{k}(x) = F(x_{k}) + (x - x_{k}) \ F^{'}(x_{k})
$$

Then, $x_{k+1}$ is a root of $p_{k}(x)$.
$$
\begin{align}
p_{k}(x_{k+1}) = 0
&\to F(x_{k}) + (x_{k+1} - x_{k}) \ F^{'}(x_{k}) = 0 \\[6pt]
&\to x_{k+1} = x_{k} - \frac{F(x_{k})}{F^{'}(x_{k})}
\end{align}
$$

---
`Convergence Guarantee` 
`Newton's Method` doesn't always converge.