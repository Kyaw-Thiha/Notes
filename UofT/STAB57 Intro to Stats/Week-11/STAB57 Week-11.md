# Week-11
Let $X_{1}, \dots, X_{n} \sim$ distribution with $f(x) = \begin{pmatrix}x_{i} - 1 \\ 2\end{pmatrix} p^{3}(1 - p)^{x-3}$.
Let prior be $\pi(p) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \ \Gamma(\beta)} \ p^{\alpha-1} \ (1-p)^{\beta-1}$.

Then, the likelihood is
$$
L(p) = \prod \begin{pmatrix}x_{i} - 1 \\ 2\end{pmatrix} p^{3}(1 - p)^{x-3}
$$
We could get that the posterior is 
$$
\pi(p \mid x_{1}, \dots, x_{n}) = L(p)
$$

---