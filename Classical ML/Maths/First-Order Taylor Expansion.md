# First-Order Taylor Expansion
#math
`Pre-Condition`
Suppose $f(x)$ is the smooth scalar function.

`First-Order Approximation`
$$
f(t + \tau) 
= f(t) + f'(t) \ \tau + O(\tau^2)
$$

`Integral Form`
Integrating the approximation.
$$
\begin{align}
&\int^{dt}_{0} f(t + \tau) \ dt \\[6pt]
&= \int^{dt}_{0} f(t) + f'(t) \ \tau + O(\tau^2)  
\ d\tau \\[6pt]
&= f(t) \ dt + \frac{1}{2} f'(t) \ dt^2 + O(dt^3)
\end{align}
$$

---
