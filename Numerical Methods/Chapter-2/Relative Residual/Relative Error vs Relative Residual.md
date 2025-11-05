# Relative Error vs Relative Residual

We have
- `Computed`: $A.\hat{x} = b - r$
- `True`: $A.x = b$

$$
\begin{aligned}
A\hat{x} - b &= r \\
- \quad A x - b &= 0 \\
\hline
A(x - \hat{x}) &= -r \iff (x - \hat{x}) = A^{-1} r
\end{aligned}
$$

Then, we get
1. $||x - \hat{x}|| = ||A^{-1}r|| \leq ||A^{-1}r$
2. $\begin{align}b &= Ax \\ ||b|| &= ||Ax|| \\ ||A|| \ ||b|| &\leq ||x|| \end{align}$

Combining 1 and 2, 
$$
\begin{align}
\frac{||x - \hat{x}||} {||A|| \ ||x||} &\leq \frac{||A^{-1}|| \ ||r||}{||b||} \\[6pt]

\iff \underbrace{\frac{||x - \hat{x}||}{||x||}}_{relative \ error}  
&\leq \underbrace{||A|| \ ||A^{-1}||}_{condition \neq cond(A)} . \underbrace{\frac{||r||}{||b||}}_{relative \ residual}
\end{align}
$$


By definition, $cond(A) = ||A|| \ ||A^{-1}||$
Can derive a lower bound for relative error using `relative error`?

I.e. $cond(A) \geq 1$  always
$$
\frac{||x - \hat{x}||}{|| x ||} \leq cond(A) \ \frac{||r||}{||b||}
$$

If $cond(A)$ is very large, the problem is `poorly conditioned`.
A small `relative residual` does not necessarily mean a small relative error.

If $cond(A)$ is not too large, the problem is `well conditioned`.
A small `relative residual` is a reliable indicator of a small relative error.

Conditioning is a continuous spectrum.
How large is "very large" depends on the context.



