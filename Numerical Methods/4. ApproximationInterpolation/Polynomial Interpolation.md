# Polynomial Interpolation
#numerical-methods/interpolation/polynomial 
`Polynomial Interpolation` is finding a polynomial $p(x)$ $s.t.$ 
$$
p(x_{i}) = F(x_{i}), \ i=0,1,2,\dots
$$

![Polynomial Interpolation](https://www.mscroggs.co.uk/img/full/runge-chebyshev.gif)

---
`Basis`
Consider $P(n)$ which is a set of polynomials of degree $\leq n$.
This is a function space, and requires the basis of $n+1$ functions.
The most common basis is the `monomial basis`, which is $\{ x^i, \quad i=0,1,2,\dots \}$

`Weierstrass' Theorem`
If a function $F$ is continuous on an interval $[a, b]$, then for any $\epsilon > 0, \exists p(\epsilon) \ s.t. \ ||F - p(\epsilon) < \epsilon||$.
In other words, for any continuous function on a closed interval $[a, b]$, there exists some polynomial that is as close to it as it can be.

---
## Basis

`1. Monomial Basis (Vandermonde Theorem)`
This is the matrix view of `Polynomial Interpolation`, that uses `monomial basis`.

For any sets $\{ x_{i}, \ i=0,1,2,\dots \}$ and $\{ y_{i}, \ i=0,1,2,\dots \}$, 
for all distinct $y_{i}$, $\exists$ a unique polynomial $P(x) \in P_{n}$ $s.t.$ $P(x_{i}) = y_{i}, \ i=0,1,2,\dots$

[[Vandermonde Theorem|Read More]]

`2. Lagrange Basis`
