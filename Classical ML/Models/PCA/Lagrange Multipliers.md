# Lagrange Multipliers
#math/lagrange 

`Lagrange Multipliers` are a method to solve constrained optimization problems.

![Lagrange Multiplier](https://mymathapps.com/mymacalc-sample/MYMACalc3/Part%20II%20-%20Derivatives/MaxMin/graphics/der_lagrange_min.gif)

---
`Motivation`

Note that having a constraint $g(x)$ remove degrees of freedom.
The problem is optimization methods like [[Gradient Descent]] are `unconstrained optimization` method.

![Lagrange Multiplier|350](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*O2nPkCDfHTsI4cJBX9rM_g.gif)

So, we need to convert it into unconstrained optimization using `Lagrange Multipliers`.

---
`Forming Lagrangian`
Let $E(x)$ be the function we are trying to optimize.
Let $g(x)=0$ be the constraint we have.

Then, we can define the `Lagrangian` objective function as
$$
L(x, \lambda) = E(x) + \lambda g(x)
$$

`Extrema`
Note that finding extrema of unconstrained $L(x, \lambda)$ is equivalent to find extrema of constrained $E(x)$
$$
\begin{align}
&\begin{cases}
\frac{\partial}{\partial x} L(x, \lambda)
= 0 \\[6pt]
\frac{\partial}{\partial \lambda} L(x, \lambda) = 0
\end{cases} \\[8pt]

\implies &\begin{cases}
\nabla E(x) + \lambda \nabla g(x) = 0 \\[6pt]
g(x) = 0
\end{cases}
\end{align}
$$

`Geometric Interpretation`
Note that at the `optimum point`,
$$
\nabla E(x) = -\lambda \nabla g(x)
$$ 
This means that gradient of the `objective function` is parallel to gradient of the `constraint surface`.

---
## See Also
- [Constrained Optimization](https://medium.com/@AyushPaniiiiii/constraint-optimization-in-svm-hard-margin-using-lagrangian-multiplier-fc7933042c40)
