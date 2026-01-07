# Single Value Decomposition
#math/linear-algebra/svd 
`SVD` is a matrix factorization technique that breaks down any $m \times n$ matrix into $A = U\Sigma V^T$.

![SVD|500](https://towardsdatascience.com/wp-content/uploads/2023/11/12D8Fth_7VQ4AL6fPV9ROxw.png)

---
`What is it?`
Every $m \times n$ matrix factors into $A = U \ \Sigma \ V^T$
where
- $U$ is an orthogonal matrix
- $\Sigma$ is a diagonal matrix
- $V^T$ is an orthogonal matrix

Geometrically, it can be thought of as $(\text{rotate})(\text{stretch})(\text{rotate})$.

Hence, we can denote it as
$$
A = 
\begin{bmatrix}
| & | & | \\
u_{1} & u_{2} & \dots \\
| & | & |
\end{bmatrix}

\begin{bmatrix}
\sigma_{1}   \\
& \ddots & &  \\
& & \sigma_{2} & \\
& & & 0 \\
& & & & \ddots \\
\end{bmatrix}

\begin{bmatrix}
- & v^T_{1} & - \\
- & v^T_{2} & - \\
 & \vdots & 
\end{bmatrix}
$$
where
- $u_{i}$ is a `left-singular vector`
- $\sigma_{i}$ is a `singular value`
- $v_{i}$ is a `right-singular vector`

---
`Connecting to eigenvalues`

Consider $A^TA$.
$$
A^TA
= (V \Sigma^T U^T) (U \Sigma V^T)
= V \Sigma^T \Sigma V^T
$$
Since $\Sigma^T \Sigma$ is diagonal, $A^TA = V\Sigma^T\Sigma V^T$ is essentially an `eigen-decomposition` with $\lambda_{i} = \sigma_{i}^2$ and $v_{i}$ being `eigen-vectors`.

Likewise, consider $AA^T$.
$$
AA^T
= (U\Sigma V^T)(V\Sigma^TU^T)
= U \Sigma \Sigma^T U^T
$$
Since $\Sigma \Sigma^T$ is diagonal, $AA^T = U\Sigma\Sigma^TU^T$ is essentially an `eigen-decomposition` with $\lambda_{i} = \sigma_{i}^2$ and $u_{i}$ being `eigen-vectors`.

---
`Example`

Consider the matrix $A = \begin{bmatrix}2 & 2 \\ 1 & 1\end{bmatrix}$.
Note that $A$ is a `singular`. 

Using `SVD`, it can be factorized as
$$
\begin{bmatrix}
2 & 2 \\
1 & 1
\end{bmatrix}
= \frac{1}{\sqrt{ 5 }}\begin{bmatrix}
2 & 1 \\
-1 & 2
\end{bmatrix}
\quad
\begin{bmatrix}
\sqrt{ 10 } \\
& 0
\end{bmatrix}
\quad
\frac{1}{\sqrt{ 2 }}
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
$$

Note that if we were to apply [[PCA Algorithm]] on $A$ to get the principle component, we would only care about
$$
\frac{1}{\sqrt{ 5 }}
\begin{bmatrix}
2 &  \\
-1 & 
\end{bmatrix}
\begin{bmatrix}
\sqrt{ 10 } & \\
& 
\end{bmatrix}
\frac{1}{\sqrt{ 2 }}
\begin{bmatrix}
1 & 1 \\
& 
\end{bmatrix}
$$
---
## See Also
- [[PCA Algorithm]]
- [[Moore-Penrose Pseudoinverse]]