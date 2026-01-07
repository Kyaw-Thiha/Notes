# Moore-Penrose Pseudoinverse
#math 
For any matrix $A \in R^{m \times n}$, the `Moore-Penrose Pseudoinverse` $A^+$ is a generalized inverse that works even when $A$ is not a square matrix or not full-rank.

---
`Pseudoinverse`
Let $A \in R^{m \times n}$ be an arbitrary matrix.
Applying [[Single Value Decomposition (SVD)|SVD]], we can get
$$
A = U \Sigma V^T
$$

Then, we can define the pseudoinverse as
$$
A^+ = V \Sigma^+ U^T
$$
where 
- $\Sigma^+$ is formed by taking `reciprocals` of non-zero singular values in $\Sigma$
- $V$ is the right singular vectors matrix
- $U$ is the left singular vectors matrix

---
`Key Properties`
- $AA^+A = A$
- $A^+AA^+ = A^+$
- $AA^+$ and $A^+A$ are symmetric.
- Always exists and is unique.

---
## See Also
- [[Single Value Decomposition (SVD)]]
