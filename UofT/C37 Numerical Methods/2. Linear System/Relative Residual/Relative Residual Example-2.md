# Example-2

`Question`
Let $A = \begin{bmatrix}c & s \\ -s & c\end{bmatrix}$ where $c^2 + s^2 = 1$
E.g: $c = \cos(\theta)$ and $s = \sin(\theta)$ for any angle $\theta$

---
`Solution`

Hence, $A^{-1} = \frac{1}{c^2 + s^2} \begin{bmatrix}c & -s \\ s & c\end{bmatrix} = \begin{bmatrix}c & -s \\ s & c\end{bmatrix} = A^T$

Note that $A^{-1} = A^T$ is an `orthogonal matrix`

`2-norm` of a matrix is square root of largest eigenvalue of the matrix.
$$||A||_{2} := \sqrt{ \sigma_{max}(A^TA) } = \sqrt{ \sigma_{max}(I) } = 1$$
and
$$
\begin{align}
&||A^{-1}||_{2} := \sqrt{ \sigma((A^{-1})^T \ (A^{-1})) } = \sqrt{ \sigma(A \ A^{-1}) } = \sqrt{ \sigma(I) } \\[6pt]

\implies  &cond_{2}(A) = ||A||_{2} ||A^{-1}||_{2} = 1
\end{align}
$$

Recall that $cond_{2}(A) = 1 \implies \text{perfectly well-conditioned}$
$$
\frac{||x - \hat{x}||_{2}} {||x||_{2}} 
\leq cond_{2}(A) \frac{||r||_{2}}{||b||_{2}}  
= \frac{||r||_{2}}{||b||_{2}}
$$

`Relative Residual` is the upper bound for `Relative Error` for a perfectly well-conditioned problem.

---
