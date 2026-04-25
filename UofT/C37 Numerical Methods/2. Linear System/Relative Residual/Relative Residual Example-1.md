# Example-1 

`Question`

Let $A = \begin{bmatrix}.780 & .563 \\ .913 & .659\end{bmatrix}$

---

`Inverse Derivation`
$$
\begin{align}
A &= \begin{bmatrix}
a & b \\ c & d
\end{bmatrix} \\[6pt]

\implies A^{-1} &= \frac{1}{ad - bc} 
\begin{bmatrix}
d & -b \\ -c & a
\end{bmatrix} \\[6pt]

\implies A^{-1} &= \frac{1}{\det(A)} 
\begin{bmatrix}
d & -b \\ -c & a
\end{bmatrix}
\end{align}
$$

Hence, $A^{-1} = 10^6 \times \begin{bmatrix} .659 & -.563 \\ -.913 & .780 \end{bmatrix}$

---

`Solution`

We want to use a norm value that is easy for us to use.

In this case, we will use the [[Infinity Norm]] which is basically the maximum absolute row sum of the matrix.

So, 
- $||A||_{\infty} = |0.913| + |0.659| =  1.572$
- $||A^{-1}||_{\infty} = |-0.953| + |0.78| = 1.693 \times 10^6$

Hence, 

$$\begin{align}cond_{\infty}(A) &= ||A||_{\infty} . ||A^{-1}||_{\infty}\\[6pt] &= 2.66 \times 10^6\end{align}$$

---
`Conclusion`
$$
\frac{||x - \hat{x}||}{||x||} 
\leq 2.66 \times 10^6 \ \frac{||r||_{\infty}} {||b||_{\infty}}
$$

The relative error in $x$ could be as much as $2.66 \times 10^6$ times the relative residual.
Hence, this is `poorly conditioned problem`

In other words, `relative residual` is not a reliable indicator of `relative error`.

---