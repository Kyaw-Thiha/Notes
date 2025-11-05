# Rate of Convergence
#numerical-methods/non-linear/rate-of-convergence 

`Definition`
If $\lim_{ \tilde{x}_{k} \to \tilde{x} } \frac{|\tilde{x} - x_{k+1}|}{|\tilde{x} - x_{k}|^p} = c \neq 0$, 
then we have `p-th order convergence` to fixed point $\tilde{x}$

`Example`
This example will show the importance of $p$.
Consider the table of absolute errors of iterations $|\tilde{x} - x_{k}|$ below.

![[Rate of Convergence.png]]
We start off with $10^{-1}$ for both systems.

For `System-1` ($p=1, \ c=\frac{1}{2}$), we get $|\tilde{x} - x_{k+1}| = \frac{|\tilde{x} - x_{k}|} {2}$.
Hence with each iteration, we divide by $2$

For `System-2` ($p=1, \ c=\frac{1}{2}$), we get $|\tilde{x} - x_{k+1}| = |\tilde{x} - x_{k}|^2$.
Hence with each iteration, we square.

Notice that despite having $c=1$, the `System-2` converge faster.
This is because of $p=2$.

---
## See Also
- [[Non-Linear Methods]]
- [[Fixed Point Iteration (FPI)]]


