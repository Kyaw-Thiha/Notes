# Vandermonde Theorem

`Vandermonde Theorem`, also known as `Method of Undetermined Coefficients`, guarantees the existence and uniqueness of the interpolating polynomial.

---
`Theorem`
For any sets $\{ x_{i} \}$ and $\{ y_{i} \}$, for every distinct $x_{i}$ and undistinct $y_{i}$, 
$\exists$ a unique polynomial $p(x) \in P_{n}$ $s.t.$ $p(x_{i}) = y_{i}, \ i=0, 1, 2, \dots$

`Proof`
If $P(x)$ exists, then it must be possible to write it as 
$$
p(x) = \sum^n_{i=0} a_{i} \ x^i
$$
This can be converted into a matrix problem with $p(x_{i}), \ i=0, 1, 2, \dots  n$.

We can solve for all the $a_{i}$ using the `Vandermonde Matrix`.

$$
\underbrace{
\begin{bmatrix}
(x_{0})^0 & (x_{0})^1 & (x_{0})^2 & \dots & (x_{0})^n  \\

(x_{1})^0 & (x_{1})^1 & (x_{1})^2 & \dots & (x_{1})^n  \\

\vdots & \vdots & \vdots &  & \vdots  \\

(x_{n})^0 & (x_{n})^1 & (x_{n})^2 & \dots & (x_{n})^n  \\
\end{bmatrix}}_{
\text{Vandermonde Matrix}
}
\begin{bmatrix}
a_{0} \\
a_{1} \\
\vdots \\
a_{n}
\end{bmatrix}
=
\begin{bmatrix}
y_{0} \\
y_{1} \\
\vdots  \\
y_{n}
\end{bmatrix}
$$

The `Vandermonde Matrix` is non-singular because all the columns are linearly independant.

The `Vendermonde Theorem` proves existence of interpolating polynomial by giving the `monomial basis`.
However, it does not lead to best algorithm since it can be poorly-conditioned.

---

