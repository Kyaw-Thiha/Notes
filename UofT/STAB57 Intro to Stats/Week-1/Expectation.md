# Expectation
[[Expectation|Expected value]](mean/average) of random variable$(X)$ is defined as
- $E[X] = \int^{\infty}_{-\infty} x \ f(x) \ dx$ when $X$ is continuous or
- $E[X] = \sum_{i} x_{i} P[X = x_{i}]$ when $X$ is discrete

---
## Properties
- [[Expectation]] is a `linear operator`.

$$
E[aX + bY + c] = a \ E[X] + b \ E[Y] + c
$$

- $E[c = c]$ where $c$ is a constant.
- $E[X]$ once calculated, is a constant $\implies E[E[X]] = E[X]$
- $E[X - E[X]] = 0$, for any $X$.

$$
\begin{align}
E[X - E[X]]
&= E[X - E[E[X]]] 
\end{align}
$$

  This is shifting the mean left(when $X>0$) or right.

> Think of [[Expectation|expectation]] as center of a circle(or center of a distribution).

---