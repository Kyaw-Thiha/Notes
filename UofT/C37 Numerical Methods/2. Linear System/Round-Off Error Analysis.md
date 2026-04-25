# Round-Off Error Analysis of GE

Factorization $PA = LU$
Because of rounding errors (both initial & propagated), we actually get $\hat{L}, \hat{U}, \hat{P}$ such that
$$
\hat{P}(A + E) = \hat{L}\hat{U}
$$

Hopefully, $||E||$ is small compared to $||A||$.

Actually, solving (forward & backward) can introduce more roundoff error, but this can still be represented as 
$$
\begin{align}
(A + \tilde{E}) \ \hat{x}  = b 
\quad & \text{ where } \tilde{E} \text{ is slightly different than } E
\end{align}
$$
Deep distinction: $(A + E) \ \hat{x} = b$
Equivalently, let $E\hat{x} = r$
$(A + \hat{r}) \ \hat{x} = b \leftrightarrow r = b - A\hat{x}$

If we use `row partial pivoting` during the factorization, we can show that 
$$
||E|| < k \ eps.||A||
$$
where $k$
- is not too large
- grows with $n$
- depends on pivoting

Similarly, for the computed solution $\hat{x}$ and the residual $r = b - A\hat{x}$

$$
||r|| < k \ eps.||b||
$$
equivalent to 
$$
\frac{||r||}{||b||} < k \ eps
$$

We need the relationship between relative error and relative residual.
In particular, when does a small relative residual (which is guaranteed if we use row partial pivoting), guarantee a small relative error?


