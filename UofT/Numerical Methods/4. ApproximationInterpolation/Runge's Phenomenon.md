# Runge's Phenomenon

`Runge's Phenomenon` is a problem of oscillation at the edges of the interval when [[Polynomial Interpolation|interpolating with polynomials]] of high degrees.

![Runge Phenomenon|400](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Runge_phenomenon.svg/960px-Runge_phenomenon.svg.png)

where
- `red line` is the function $y(x) = \frac{1}{1 + 25x^2}$
- `blue line` is the fifth-order polynomial intepolation 
  (interpolating at $6$ points)
- `green line` is the ninth-order polynomial interpolation
  (interpolating at $10$ points)

---
`Visualization`
Note the edges of the polynomials.

![Runge's Phenonpmenon Visualization](https://blogs.mathworks.com/cleve/files/n_animation.gif)

---
`Main Idea`

`Runge's Phenomenon` shows that more interpolation points with higher degrees can lead to worse fit.
Hence, it's dangerous to interpolate many points with a single high degree polynomial.

Instead, use [[Piecewise Interpolation]].

---
`Relation to Weierstrass Theorem`

By [[Polynomial Interpolation|Weierstrass Theorem]], it might seem that interpolating with higher degree polynomial would leads to a better fit.

However, `Weierstrass Theorem` only states that a set of polynomial function exists without giving specific method of finding it.
Hence, $P_{n}(x)$ is not guaranteed to have property of `uniform convergence`.

---
