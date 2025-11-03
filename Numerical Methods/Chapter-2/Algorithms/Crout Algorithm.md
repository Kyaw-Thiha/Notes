# Crout Algorithm
#numerical-methods/linear-systems/gaussian-elimination/crout

Crout algorithm is an algorithm to carry out LU factorization for Gaussian Elimination.

The main idea is to build lower triangular $L$ one row at a time and upper triangular $U$ one column at a time.

## Idea
Consider a $3 \times 3$ matrix $A$ such that
$$
A = 
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0 \\
l_{21} & 1 & 0 \\
l_{31} & l_{32} & 1 \\
\end{bmatrix}
.
\begin{bmatrix}
u_{11} & u_{12} & u_{13} \\
0 & u_{22} & u_{23} \\
0 & 0 & u_{33} \\
\end{bmatrix}
$$
Then at each loop step, Crout Algorithm will be finding the $l_{ij}$ and $u_{ji}$.

### General Formulas (Crout Algorithm)
At step $i$:

1. Compute column $i$ of $L$:
$$
l_{ji} = a_{ji} - \sum_{k=1}^{i-1} l_{jk} \, u_{ki}, \quad j \geq i
$$

2. Set the diagonal of $U$ to $1$:
$$
u_{ii} = 1
$$

3. Compute row $i$ of $U$:
$$
u_{ij} = \frac{1}{l_{ii}} \left( a_{ij} - \sum_{k=1}^{i-1} l_{ik} \, u_{kj} \right), \quad j > i
$$

## Comparism to Crout Algorithm
- **Doolittle:** 
  - builds row $i$ of $U$ first, then column $i$ of $L$.
  - $L$ has **1s on the diagonal**, and $U$ holds the pivots.
  
- **Crout:** 
  - builds column $i$ of $L$ first, then row $i$ of $U$.
  - $U$ has **1s on the diagonal**, and $L$ holds the pivots.

## Minimal Algorithm
```python
def crout_lu(A):
    """
    Crout LU Decomposition (no pivoting).
    Produces A = L * U, with diag(U) = 1.
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        # --- Step 1: compute column i of L ---
        for j in range(i, n):
            L[j][i] = A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))
        
        # --- Step 2: set diagonal of U to 1 ---
        U[i][i] = 1.0
        
        # --- Step 3: compute row i of U ---
        for j in range(i + 1, n):
            U[i][j] = (A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))) / L[i][i]
    
    return L, U
```

## See Also
- [[Gaussian Elimination]]
- [[Solving Linear Systems]]
- [[Dolittle Algorithm]]