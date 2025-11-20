# Round-Off Error

Solving `forward & backward substitution` can introduce more roundoff error, but this can still be represented as:
$$
(A + \tilde{E}) \ \hat{x} = b \quad \text{where } \tilde{E} \text{ is slightly different from } E
$$

Equivalently, let $E\hat{x} = r$
$$
(A + \bar{C}) \hat{x} = b \iff \underbrace{r = b - Ax}_{residual}
$$

If we use row partial pivoting during factorization, we can show that
$$
||E|| < k.eps ||A||
$$
where $k$ is not too large
- grows with $n$
- depends on pivoting

Similarly, for the computed solution $\hat{x}$ and the residual $r  =b - A\hat{x}$
$$
||r|| < k.eps||b||
$$
