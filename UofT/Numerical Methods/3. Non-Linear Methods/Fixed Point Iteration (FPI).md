# Fixed Point Iteration (FPI)
#numerical-methods/non-linear/fpi 

Start with an approximate solution $\hat{x}_{0}$, then iterate $\hat{x}_{k+1} = g(\hat{x}_{k})$, $k = 0, 1, 2, \dots$ until convergence or failure.

---

`Example`
Let $F(x) = x^2 + 2x - 3$. 
We know that the roots are $1$ and $-3$.

Consider the `FPI` $x_{k+1} = x_{k} + \frac{(x_{k})^2 + 2x - 3} {(x_{k})^2 - 5}$
for the `Fixed Point Problem` $x = g(x) = x + \frac{x^2 + 2x -3}{x^2 - 5}$

We see that this is the [[Fixed Point Methods (FPM)#Second Form|Second Form]] of the `Fixed Point Problem`, where $h(x) = -\frac{1}{x^2 -5}$.

---
Since $h(x) \neq 0, \forall x \in R$, we don't need to check if a fixed point is a root.

- If we start with $\hat{x}_{0} = -5$, $\hat{x}_{k}$ approach $-3$.
- If we start with $\hat{x}_{0} = 5$, $\hat{x}_{k}$ do not converge.
- If we start with $\hat{x}_{0} = 0$, $\hat{x}_{k}$ approach $0$.

Hence, we can see that depending on initial $\hat{x}_{0}$, the `FPI` may converge to some fixed point or may not converge.


## See Also
- [[Non-Linear Methods]]
- [[Fixed Point Methods (FPM)]]
