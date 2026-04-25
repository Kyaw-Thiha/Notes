# System of Non-Linear Equation

`Problem`: Solve $F(\bar{x}) = \bar{a}$

`Solution`
Extend [[Newton's Method]] for a system of non-linear equations.
$$
\bar{x}_{k+1} = \bar{x}_{k} - \frac{F(\bar{x}_{k})}{F'(\bar{x}_{k})}
$$

where $F'(\bar{x}_{k})$ is the `Jacobian Matrix` of $F$

Hence, we get
$$
\begin{align}
\bar{x}_{k+1}
&= \bar{x}_{k} - F'(\bar{x}_{k})^{-1} \ F(\bar{x}_{k}) \\[6pt]

\bar{x}_{k+1} - \bar{x}_{k}
&= - F'(\bar{x}_{k})^{-1} \ F(\bar{x}_{k}) \\[6pt]

\underbrace{F'(\bar{x}_{k}) . (\bar{x}_{k+1} - \bar{x}_{k})}_{\text{Ax}}
&= - \underbrace{F(\bar{x}_{k})}_{\text{b}} \\[6pt]
\end{align}
$$

We can see that the derived $F'(\bar{x}_{k}) . (\bar{x}_{k+1} - \bar{x}_{k})- F(\bar{x}_{k})$ is in the form of $Ax=b$.

Directly finding an inverse if very expensive.

So instead, we can use a pseudo-[[Newton's Method]] by holding the `Jacobian Matrix` fixed for a few iterations.
This means that we can reuse the $PA = LU$ factorization.
We just need to make sure that we are still converging.

---
