# Finding the Orthogonal Matrix

`Finding Eigenvalues`

From [[Centering Coordinate System]], recall that we get the covariance matrix of 
$$
S = \begin{bmatrix}
16 & 12 \\
12 & 9
\end{bmatrix}
= \begin{bmatrix}
\sigma_{1}^2 & \sigma_{1}\sigma_{2} \\
\sigma_{2}\sigma_{1} & \sigma_{2}^2
\end{bmatrix}
$$

Using this, we can get that the `total variance` of the dataset is $16 + 9 = 25$.
This means that the `sum of eigenvalues`$=25$.

Hence, we can use this `lemma` [[Eigenstructure of Symmetric Matrices]] to get that 
- `Eigenvalues` are $\lambda_{1}=25$ and $\lambda_{2} = 0$.
- `Eigenvector` of $\lambda_{1}=25$ is $v = \begin{bmatrix}4 \\ 3\end{bmatrix}$. 

---
`Forming Orthogonal Matrix`

`1. Eigenvector for non-zero Eigenvalue`
Using [[Eigenstructure of Symmetric Matrices]], we already get that 
$$
v = \begin{bmatrix}
4 \\ 3
\end{bmatrix}
, \quad 
||v|| = \sqrt{ 4^2 + 3^2 } = 5
$$

By `Spectral Theorem`, a `symmetric matrix` must have an orthonormal eigenbasis.
$$
v_{1} 
= \frac{v}{||v||}
= \frac{1}{5} 
\begin{bmatrix}
4 \\ 3
\end{bmatrix}
= \begin{bmatrix}
\frac{4}{5} \\ \frac{3}{5}
\end{bmatrix}
$$

`2. Eigenvector for zero Eigenvalue`
From [[Eigenstructure of Symmetric Matrices]], we know that the other eigenvalues are all zero.

Hence, we need a vector in $v^{\perp}$ such that
$$
v^Tw = 4w_{1} + 3w_{2} = 0
$$

A straightforward solution is 

$$
w = \begin{bmatrix}-3 \\ 4\end{bmatrix},
\quad
||w|| = \sqrt{ (-3)^2 + 4^2 } = 5
$$

By `Spectral theorem` on symmetric matrices, we need to normalize our eigenvector.
$$
v_{2} = \frac{1}{5}
\begin{bmatrix}
-3 \\ 4
\end{bmatrix}
= \begin{bmatrix}
-\frac{3}{5}  \\
\frac{4}{5}
\end{bmatrix}
$$

`3. Orthogonal Matrix`
By the `Spectral Theorem`, placing the orthonormal eigenvectors as columns gives an orthogonal matrix
$$
V = \begin{bmatrix}
v_{1} & v_{2}
\end{bmatrix}
= \begin{bmatrix}
\frac{4}{5} & -\frac{3}{5}  \\
\frac{3}{5} & \frac{4}{5}
\end{bmatrix}
$$

