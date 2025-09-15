# Dolittle Algorithm
#numerical-methods/linear-systems/gaussian-elimination/dolittle

Dolittle algorithm is an algorithm to carry out LU factorization for Gaussian Elimination.

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
Then at each loop step, Dolittle Algorithm will be finding the $l_{ij}$ and $u_{ji}$.

## General Algorithm 
At step $i$:
1. Compute row $i$ of $U$:
$$
u_{ij} = a_{ij} - \sum_{k=1}^{i-1} l_{ik} u_{kj}, \quad j \geq i
$$

2. Compute column $i$ of $L$:
$$
l_{ji} = \frac{1}{u_{ii}} \left( a_{ji} - \sum_{k=1}^{i-1} l_{jk} u_{ki} \right), \quad j > i
$$

3. Set the diagonal of $L$ to $1$:
$$
l_{ii} = 1
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
"""
Minimal Doolittle LU Decomposition (no pivoting)

Goal: factor a square matrix A into L * U
 - L = lower triangular with 1s on the diagonal
 - U = upper triangular

This is the simplest form: 
  * no pivoting
  * small, clear steps
  * pure Python lists
"""

def doolittle_lu(A):
    n = len(A)
    
    # Create empty L and U
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    
    # Loop over columns/rows
    for i in range(n):
        # --- Step 1: compute row i of U ---
        for j in range(i, n):
            # u_ij = a_ij - sum_{k=0..i-1} l_ik * u_kj
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        
        # --- Step 2: set diagonal of L to 1 ---
        L[i][i] = 1.0
        
        # --- Step 3: compute column i of L ---
        for j in range(i + 1, n):
            # l_ji = (a_ji - sum_{k=0..i-1} l_jk * u_ki) / u_ii
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    
    return L, U
```

## Full Algorithm
```python
# doolittle_lu.py
from __future__ import annotations
import numpy as np

class SingularMatrixError(Exception):
    pass

def lu_doolittle(A: np.ndarray, *, pivoting: bool = True, tol: float = 1e-12):
    """
    Doolittle LU factorization with optional partial pivoting.

    Factors A into P @ A = L @ U where:
      - L is unit lower-triangular (diag(L) = 1)
      - U is upper-triangular
      - P is a permutation matrix (identity if pivoting=False)

    Parameters
    ----------
    A : (n, n) array_like
        Coefficient matrix (will be copied as float).
    pivoting : bool, default True
        Use partial pivoting (recommended for stability).
    tol : float, default 1e-12
        Pivot tolerance to detect near-singularity.

    Returns
    -------
    P, L, U : np.ndarray
        Matrices satisfying P @ A ≈ L @ U.

    Raises
    ------
    SingularMatrixError
        If a zero (or near-zero) pivot is encountered.
    """
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    L = np.eye(n)
    U = np.zeros_like(A)
    P = np.eye(n)

    for i in range(n):
        # --- Partial pivoting on column i ---
        if pivoting:
            p = i + np.argmax(np.abs(A[i:, i]))
            if np.abs(A[p, i]) < tol:
                raise SingularMatrixError(f"Near-zero pivot at column {i}")
            if p != i:
                # Swap rows in A and P
                A[[i, p], :] = A[[p, i], :]
                P[[i, p], :] = P[[p, i], :]
                # IMPORTANT: swap the already-built part of L (columns < i)
                if i > 0:
                    L[[i, p], :i] = L[[p, i], :i]
        else:
            if np.abs(A[i, i]) < tol:
                raise SingularMatrixError(f"Zero pivot at column {i} without pivoting")

        # --- Build row i of U (j >= i) ---
        # U[i, j] = A[i, j] - sum_{k=0..i-1} L[i, k] * U[k, j]
        for j in range(i, n):
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]

        # --- Build column i of L (j > i) ---
        # L[j, i] = (A[j, i] - sum_{k=0..i-1} L[j, k] * U[k, i]) / U[i, i]
        if np.abs(U[i, i]) < tol:
            raise SingularMatrixError(f"Zero pivot produced at ({i},{i})")

        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]

    return P, L, U


def forward_sub(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L y = b for y, where L is unit lower-triangular."""
    L = np.asarray(L)
    b = np.asarray(b, dtype=float)
    n = L.shape[0]
    y = np.zeros_like(b, dtype=float)

    if b.ndim == 1:
        for i in range(n):
            y[i] = b[i] - L[i, :i] @ y[:i]  # diag(L)=1
    else:
        # multiple RHS
        m = b.shape[1]
        for i in range(n):
            y[i, :] = b[i, :] - L[i, :i] @ y[:i, :]
    return y


def back_sub(U: np.ndarray, y: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Solve U x = y for x, where U is upper-triangular."""
    U = np.asarray(U)
    y = np.asarray(y, dtype=float)
    n = U.shape[0]
    x = np.zeros_like(y, dtype=float)

    if y.ndim == 1:
        for i in range(n - 1, -1, -1):
            if np.abs(U[i, i]) < tol:
                raise SingularMatrixError(f"Zero pivot on U[{i},{i}] in back substitution")
            x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    else:
        m = y.shape[1]
        for i in range(n - 1, -1, -1):
            if np.abs(U[i, i]) < tol:
                raise SingularMatrixError(f"Zero pivot on U[{i},{i}] in back substitution")
            x[i, :] = (y[i, :] - U[i, i + 1:] @ x[i + 1:, :]) / U[i, i]
    return x


def solve(A: np.ndarray, b: np.ndarray, *, pivoting: bool = True) -> np.ndarray:
    """
    Solve A x = b using Doolittle LU with partial pivoting.
    Supports one or many right-hand sides.
    """
    P, L, U = lu_doolittle(A, pivoting=pivoting)
    b_permuted = P @ np.asarray(b, dtype=float)
    y = forward_sub(L, b_permuted)
    x = back_sub(U, y)
    return x


if __name__ == "__main__":
    # --- Demo & quick self-check ---
    np.set_printoptions(precision=4, suppress=True)

    A = np.array([[2., 1., 1.],
                  [4., -6., 0.],
                  [-2., 7., 2.]])
    b = np.array([5., -2., 9.])

    P, L, U = lu_doolittle(A)
    print("P=\n", P)
    print("L=\n", L)
    print("U=\n", U)
    print("Check PA ≈ LU:\n", P @ A - L @ U)

    x = solve(A, b)
    print("x =", x)
    print("Residual ||Ax - b|| =", np.linalg.norm(A @ x - b))

```

## See Also
- [[Gaussian Elimination]]
- [[Solving Linear Systems]]
- [[Crout Algorithm]]