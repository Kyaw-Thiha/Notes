# Newton Derivatives Example

`Relation between divided differences and derivatives`

Consider $Y[x_{1}, x_{0}] = \frac{Y(x_{1}) - Y(x_{0})}{x_{1} - x_{0}}$
$$
\begin{align}
\lim_{ x_{1} \to x_{0} } Y[x_{1}, \ x_{0}]
&= \lim_{ x_{1} \to x_{0} } \frac{Y(x_{1}) - Y(x_{0})}{x_{1} - x_{0}} \\[6pt]
&= Y'(0)\quad ,\text{ provided that } Y'(x_{0}) \text{ exists}
\end{align}
$$

Consider $Y[x_{2}, \ x_{1},\ x_{0}] = \frac{Y[x_{2}, x_{1}] - Y[x_{1}, x_{0}]}{x_{2} - x_{1}}$.

$$
\lim_{ \substack{x_{2} \to x_{0} 
\\ x_{1} \to x_{0}}  } 
Y[x_{2}, x_{1}, x_{0}]
= \frac{Y''(x_{0})}{2!}
$$

In general, we can show that 
$$
\lim_{ \substack{x_{k} \to x_{0} 
\\ x_{k-1} \to x_{0} \\ \vdots \\  x_{1} \to x_{0}} } 
Y[x_{k}, \dots, x_{0}]
= \frac{y^{(k)}(x_{0})}{k!}
$$

---

`Question`
Find $p(x) \in P_{4}$ $s.t.$ 
- $p(0) = 0$
- $p(1) = 1$, $p'(1) = 1$, $p''(1) = 2$
- $p(2) = 6$

---
`Solution`

Using $\lim_{ x_{1} \to x_{0} } y[x_{1}, \ x_{0}] = \frac{y'(x_{0})}{1!}$ and $\lim_{ \substack{x_{2} \to x_{0} \\ x_{1} \to x_{0}}  }  Y[x_{2}, x_{1}, x_{0}]= \frac{Y''(x_{0})}{2!}$,

![[Newton Derivative Example.png]]

Hence, 
$$
\begin{align}
p(x) &= \
y[x_{0}] + (x) \ y[x_{1}, x_{0}]  
+ x(x-1) \ y[x_{1}, x_{1}, x_{0}] \\[6pt]
&+ x(x-1)^2 \ y[x_{1}, x_{1}, x_{1}, x_{0}] \\[6pt]
&+ x(x-1)^3 \ y[x_{2}, x_{1}, x_{1}, x_{1}, x_{0}] \\[10pt]
&= x + x(x-1)^2 + x(x-1)^3
\end{align}
$$

---
## See Also
- [[Newton Basis]]
- [[Newton Example]]
