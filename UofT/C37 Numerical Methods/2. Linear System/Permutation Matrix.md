## Permutation Matrix
#math

A **permutation matrix** is a square matrix obtained by permuting the rows of the identity matrix.

### Definition
- A permutation matrix has exactly **one 1 in each row and column**, with all other entries equal to 0.
- Formally, if $\pi$ is a permutation of $\{1, 2, ..., n\}$, then the permutation matrix $P$ corresponding to $\pi$ has entries:
$$
P_{ij} =
\begin{cases}
1 & \text{if } i = \pi(j) \\
0 & \text{otherwise}
\end{cases}
$$

### Row and Column Permutations
- **Left multiplication**: $P A$ permutes the **rows** of $A$ according to the permutation $\pi$.
- **Right multiplication**: $A P$ permutes the **columns** of $A$.

### Example
Suppose we want to swap row 1 and row 3 of a 3×3 matrix.  
The permutation matrix is
$$
P =
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 0
\end{bmatrix}
$$

If
$$
A =
\begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{bmatrix},
$$
then
$$
P A =
\begin{bmatrix}
g & h & i \\
d & e & f \\
a & b & c
\end{bmatrix}
$$

### Properties
- Permutation matrices are **orthogonal**: $P^{-1} = P^T$.
- Determinant of $P$ is either $+1$ or $-1$, depending on whether the permutation is even or odd.
- Used in **LU decomposition with partial pivoting**: row swaps during elimination can be represented as multiplying by a permutation matrix.

### Summary
A permutation matrix is just the identity matrix with its rows shuffled. It provides a clean algebraic way to represent row/column swaps in linear algebra, especially in numerical algorithms like Gaussian elimination.

## See Also
- [[Gaussian Elimination]]
