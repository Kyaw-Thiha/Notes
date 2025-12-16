# Approximation Methods
#numerical-methods/approximation
`Approximation` aims to find a simpler function that closely matches the data.
It often ignore some data points to minimize the error.

---
## Truncated Taylor Series
$$
p(x)
= \underbrace{F(a) + F'(a)(x-a) + \dots + \frac{ F^{(n)}(a) }{n!}(x-a)^n}_{
\text{Only the first n+1 terms}
}
$$
This is the polynomial because of the $(x-a)^i$, where $i=0, 1, 2, \dots$

The error is 
$$
\begin{align}
e(x) &= p(x) - F(x) \\[6pt] 
&= \frac{F^{(n+1)}(n) }{(n+1)!} (x-a)^{n+1}
\end{align}
$$
---
## Polynomial Interpolation
Find the polynomial $p$ $s.t.$
$$
p(x_{i}) = F(x_{i}) , \quad i = 0, 1, 2, \dots
$$
where $F$ is the function we are trying to approximate.

---
## Least Squares
Find the polynomial $p$ $s.t.$ $p(x)$ minimize
$$
||F-p||_{2} = \left( \int^b_{a} (F(x) - p(x))^2 dx \right)^{1/2}
$$

We can also use other norms like
- [[Infinity Norm]]: $||F - p||_{\infty} = \max_{a \leq x \leq b} |F(x) - p(x)|$
- [[1-Norm]]: $||F - p||_{1} = \int^b_{a} |F(x) - p(x)| dx$

---
`Advice`

If you want to approximate a function around a given point and you have access to the derivatives of the function, then you may want to use `Taylor Expansion`.

If you want to approximate a function on an interval where you can access some function values but not the derivatives, you can use [[Polynomial Interpolation]].

---
## See Also
- [[Polynomial Interpolation]]
- [[Approximation and Interpolation]]