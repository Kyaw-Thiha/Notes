# Gaussian Elimination
#math #numerical-methods/linear-systems/gaussian-elimination

**Gaussian Elimination** is the most used algorithm to solve a linear equation.
Essentially,suppose we are solving for $A.x = b$.

1. First, use forward elimination to denote $A = L.U$
   where
   - $L$ is the strictly lower triangle
   - $U$ is the upper triangle

2. Secondly substituting $A=L.U$, we get
- $L.U.x = b$

3. Thirdly, since they are now triangle matrix, we can easily solve them in two steps:
- Forward Substitution: $L.d = b$
$$
\begin{bmatrix}
1 & 0 & 0 \\
l_{1} & 1 & 0 \\
l_{2} & l_{3} & 1 \\
\end{bmatrix}
.
\begin{bmatrix}
d_{1} \\
d_{2} \\
d_{3} \\
\end{bmatrix}
=
\begin{bmatrix}
b_{1} \\
b_{2} \\
b_{3} \\
\end{bmatrix}
$$
- Backward Substitution: $U.x = d$
$$
\begin{bmatrix}
u_{1} & u_{2} & u_{3} \\
0 & u_{4} & u_{5} \\
0 & 0 & u_{6} \\
\end{bmatrix}
.
\begin{bmatrix}
x_{1} \\
x_{2} \\
x_{3} \\
\end{bmatrix}
=
\begin{bmatrix}
d_{1} \\
d_{2} \\
d_{3} \\
\end{bmatrix}
$$

## Forward Elimination
Suppose we have a matrix $A$.
$$
A = 
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
$$

1. Start with $j = 1$.
   For every row $i > 1$, compute the multiplier $m_{ij} = \frac{a_{ij}}{a_{jj}}$
2. Zero out the terms below the diagonal by 
   $\text{Row}_{i} = \text{Row}_{i} - m_{ij}.\text{Row}_{j}$.
3. Store the multiplier $m_{ij} = \frac{a_{ij}}{a_{jj}}$ inside $L$ in their corresponding position
4. Repeat the process for all terms below the diagonal.
5. Store the remaining matrix as $U$.

Note that this means that
$$
L = 
\begin{bmatrix}
1 & 0 & 0 \\
m_{21} & 1 & 0 \\
m_{31} & m_{32} & 1
\end{bmatrix}
$$
and
$$
U = 
\begin{bmatrix}
u_{11} & u_{12} & u_{13} \\
0 & u_{22} & u_{23} \\
0 & 0 & u_{33}
\end{bmatrix}
$$
or more explicitly,
$$
U =
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
0 & a_{22} - m_{21}a_{12} & a_{23} - m_{21}a_{13} \\
0 & 0 & a_{33} - m_{31}a_{13} - m_{32}(a_{23} - m_{21}a_{13})
\end{bmatrix}
$$

---
### Example
Consider the following matrix

$$
A =
\begin{bmatrix}
2 & 3 & 1 \\
4 & 7 & 7 \\
-2 & 4 & 5
\end{bmatrix}
$$

#### 1. Eliminate below the pivot $a_{11}=2$

Multipliers:
- $m_{21} = \tfrac{4}{2} = 2$
- $m_{31} = \tfrac{-2}{2} = -1$


$$
U = 
\begin{bmatrix}
2 & 3 & 1 \\
0 & 1 & 5 \\
0 & 7 & 6
\end{bmatrix}
$$

#### Step 2: Eliminate below the pivot $a_{22}=1$

Multiplier:
- $m_{32} = \tfrac{7}{1} = 7$

Apply elimination:

$$
U =
\begin{bmatrix}
2 & 3 & 1 \\
0 & 1 & 5 \\
0 & 0 & -29
\end{bmatrix}
$$

#### Step 3: Collect multipliers into $L$

$$
L =
\begin{bmatrix}
1 & 0 & 0 \\
2 & 1 & 0 \\
-1 & 7 & 1
\end{bmatrix}
$$

#### Final Factorization

$$
A = L U
$$

where

$$
L =
\begin{bmatrix}
1 & 0 & 0 \\
2 & 1 & 0 \\
-1 & 7 & 1
\end{bmatrix}, \quad
U =
\begin{bmatrix}
2 & 3 & 1 \\
0 & 1 & 5 \\
0 & 0 & -29
\end{bmatrix}
$$

---
## Pivot
Note that when calculating multipliers $m_{ij} = \frac{a_{ij}}{a_{jj}}$, 
- if the diagonal term $a_{jj} = 0$, we can't divide
- or if the diagonal term $a_{jj} = \epsilon$, the result will explode 

In those cases, carry out one of following methods.
- **Minimal Pivoting**: take the next term $a_{(j+k)j} \neq 0$.
- **Partial Pivoting**: take the largest term $a_{ij}$ in column $j$
- **Complete Pivoting**: take the largest term $a_{ij}$ in the entire matrix

Note that although **Complete Pivot** offers a higher numerical stability at the cost of performance, **Partial Pivot** is enough for most practical applications.

## Partial Pivot
To guarantee numerical stability, we need to fulfill the condition that $|m_{ik}| \leq 1$.
The way to fulfill that condition is by taking the largest $a_{ij}$ in the column $j$.

This means that we need to use [[Permutation Matrix]] to swap the highest $a_{ij}$ with the diagonal term $a_{jj}$.

Let $A$ be the matrix we are trying to solve.
Then $M$ is the elimination operator such that 
- $M = M_{n-1}.P_{n_-1} \dots M_{1}.P_{1}$   
- $U = M.A$

Thus, we can denote the permutations carried out in each steps as $P = P_{n-1}\dots P_{1}$ where $P$ is the multiplications of all permutation matrices.

This essentially mean that we are using
$$
PA = LU
$$
where 
- $P$ is the multiplication of row permutation matrices
- $L$ is the strictly lower triangular matrix
- $U$ is the upper triangular matrix

To substitute in back to the original $Ax = b$,
$$
A = P^{-1}.LU
$$
where 
- $P^{-1} = P^T$ since permutation matrices are orthogonal

## Complete Pivot
Compared to partial pivot, complete pivot takes the largest $a_{ij}$ in the entire matrix $A$ at each step.

This yields a more numerically stable calculations, but require higher computation.

This essentially mean that we are solving the equation
$$
PAQ = LU
$$
where 
- $P$ is the multiplication of row permutation matrices
- $Q$ is the multiplication of column permutation matrices

To substitute in back to the original $Ax = b$,
$$
A = P^{-1}.LU.Q^{-1}
$$
where 
- $P^{-1} = P^T$ since permutation matrices are orthogonal
- $Q^{-1} = Q^T$ since permutation matrices are orthogonal

---
## See Also
- [[Permutation Matrix]]
- [[Solving Linear Systems]]