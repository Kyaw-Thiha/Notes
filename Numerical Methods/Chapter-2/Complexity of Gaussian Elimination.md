# Complexity of Gaussian Elimination

We will count multiplication/addtion pairs
i.e. $mx + b$       a `FLOP` (floating point operation)

This `FLOP` is the most expensive operation (much more expensive than comparism), so we are counting `FLOPs` only.
Adding multiple of row-1 to rows 2, ..., n costs -> (n-1)%2cR

For the second stage, $(n-2)^2$ FLOPs

For $(n-1)^2 + (n-2)^2 + \dots + 1^2 \text{ FLOPs} = \frac{n(n-1)(2n-1)} {6} = \frac{n^3}{2} + O(n^2)$


Total forward/backward solve complexity:
$$
n^2 + O(n) \text{ FLOP}
$$
It takes $\frac{n}{3}$ times cheaper to factorize than to solve.

If you have several systems to solve with same coefficient matrix $A$ but different right hand sides.
$$
\begin{align}
Ax = b  \\
Ax = c \\
Ax = e \\
Ax = f \\
\dots
\end{align}
$$
it takes $\frac{n}{3}$ times cheaper to factor $A$ once only and then use factorization for each right hand side.