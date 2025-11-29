# PCA Algorithm
#ml/models/pca 

`Notations`

Given: $D=\{ y_{i} \}_{i=1}^N$ where $y_{i} \in R^d, \forall i$
Goal: $\{ x_{i} \}^N_{i=1}$, where $x_{i} \in R^d, \forall i$

---
`PCA Algorithm`
1. Compute the `sample mean`.
$$
\mu = \frac{1}{N} \sum^{N}_{i=1} y_{i}
$$

2. Compute the `sample variance`.
$$
\begin{align}
S  
&= \frac{1}{N-1} \sum^{N}_{i=1} (y_{i} - \mu) \\[6pt]
&= \frac{1}{N-1} (Y - \mu .1) 
(Y - \mu.1)^T \\[6pt]
\end{align}
$$
where $Y = \begin{bmatrix}y_{1} & \dots & y_{N}\end{bmatrix}$ and $1_{d} = \begin{bmatrix} 1 & \dots & 1\end{bmatrix}$

3. Perform `Eigen-Decomposition` $S$

$$
\begin{align}
S &= V \land V^T \\[6pt]
&= \begin{bmatrix}
| & & | \\
V_{1} & \dots & V_{d} \\
| & & | \\
\end{bmatrix}
\begin{bmatrix}
\lambda_{1} & &  \\
& \dots &  \\
& & \lambda_{d}
\end{bmatrix}
\begin{bmatrix}
- & v_{1}^T & -  \\
& \vdots &  \\
- & v_{d}T & - 
\end{bmatrix}
\end{align}
$$
where $VV^T = V^TV = Id$

4. Sort the `eigenvalues`.
$$
\lambda_{1} \geq \lambda_{2} \geq \dots \geq \lambda_{d}
$$

5. Choose $top-k$ `eigenvalues`, and make the subspace out of the corresponding `eigenvectors`.
$$
W = \begin{bmatrix}
v_{1} & v_{2} & \dots & v_{k}
\end{bmatrix}
$$
where $\lambda_{1} \geq \lambda_{2} \geq \dots \geq \lambda_{k} \geq \dots \geq \lambda_{d}$

6. Project with the new subspace.
$$
x_{i} = W^T (y_{i} - \mu)
$$

---
`Remarks on Variance`
- Total Variance of data: $\sum^{d}_{i=1} \lambda_{i}$
- Variance in principal component subspace: $\sum^{k}_{i=1} \lambda_{i}$
- Variance lost: $\sum^{d}_{i=k+1} \lambda_{i}$
