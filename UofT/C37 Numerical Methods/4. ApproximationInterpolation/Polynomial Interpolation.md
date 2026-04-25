# Polynomial Interpolation
#numerical-methods/interpolation/polynomial 

`Polynomial Interpolation` is finding a polynomial $p(x)$ $s.t.$ 
$$
p(x_{i}) = F(x_{i}), \ i=0,1,2,\dots
$$

![Polynomial Interpolation|400](https://www.mscroggs.co.uk/img/full/runge-chebyshev.gif)

---
`Basis`
Consider $P(n)$ which is a set of polynomials of degree $\leq n$.
This is a function space, and requires the basis of $n+1$ functions.
The most common basis is the `monomial basis`, which is $\{ x^i, \quad i=0,1,2,\dots \}$

`Weierstrass' Theorem`
If a function $F$ is continuous on an interval $[a, b]$, then for any $\epsilon > 0, \exists p(\epsilon) \ s.t. \ ||F - p(\epsilon) < \epsilon||$.
In other words, for any continuous function on a closed interval $[a, b]$, there exists some polynomial that is as close to it as it can be.

Alternatively,
$$
\lim_{ n \to \infty } \left( \sup_{a\leq x\leq b} 
|f(x) - P_{n}(x)| \right) = 0
$$

---
## Basis

`1. Monomial Basis (Vandermonde Theorem)`
First, represent the interpolating polynomial in the standard monomial basis
$$
p(x) = a_{0} + a_{1}x + \dots + a_{n} x^n
$$
Then, determine their coefficients by solving a linear system.

We can solve this linear system by forming a `Vandermonde Matrix`
$$
V_{ij} = x_{i}^j
$$
If all $x_{i}$ are distinct, then $\det(V) \neq 0$.
So, the system has a unique solution.

[[Vandermonde Theorem|Read More]]
[[Vandermonde Example|See Example]]

$$

$$

`2. Lagrange Basis`
We attempt to construct a basis of polynomials that are $1$ at one node, and $0$ at all others.

We can define the basis function as
$$
l_{i}(x) 
= \prod_{j\neq i} \frac{x - x_{j}}{x_{i} - x_{j}}
\quad \text{ where } l_{i}(x_{j}) = \delta_{ij}
$$
Then, we can get the `interpolant` as
$$
p(x) = \sum^n_{i=0} y_{i} \ l_{i}(x)
$$

[[Lagrange Matrix|Read More]]
[[Lagrange Example|See Example]]

$$

$$

`3. Newton Basis`
The main idea is to build the interpolating polynomial incrementally, adding one data at a time.
$$
p(x) = a_{0} + a_{1}(x-x_{0}) 
+ a_{2}(x-x_{0})(x-x_{1}) + \dots
$$
where the coefficients $a_{i}$ are divided differences
$$
a_{k} = f[x_{0}, \ x_{1}, \ \dots, \ x_{k}]
$$

[[Newton Basis|Read More]]
[[Newton Example|See Example]]

---
## See Also
- [[Vandermonde Theorem]]
- [[Lagrange Matrix]]
- [[Newton Basis]]
- [[Runge's Phenomenon]]
- [[Error in Polynomial]]
- [[Piecewise Interpolation]]

