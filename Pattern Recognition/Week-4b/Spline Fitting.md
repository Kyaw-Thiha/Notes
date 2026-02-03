# Spline Fitting
`Spline fitting` fits a [[Polynomial Fitting|polynomial]] between each sequence of `knot points`.

![Spline Fitting|400](https://blogs.sas.com/content/iml/files/2020/05/cubicInterp1.png)

---
## Main Idea

`Splines polynomial functions` bridge a sequence of `knot points` $(x_{0}, y_{0}), \ \dots, \ (x_{k}, y_{k})$ such that 
$$
\hat{f}(x; \ \theta)
= \begin{cases}
q_{1}(x) & \text{if } x_{0} < x < x_{1} \\
q_{2}(x) & \text{if } x_{1} < x < x_{2} \\
\vdots \\
q_{k}(x) & \text{if } x_{k-1} < x < x_{k} \\
\end{cases}
$$

with `continuity constraints` at the knot points:
- $q_{i}(x_{i}) = q_{i+1}(x_{i}) = y_{i}$
- $\frac{d^n}{dx^n} [q_{i} (x_{i})] = \frac{d^n}{dx^n} [q_{i+1}(x_{i})], \ \forall n < deg(q_{i})$

---
