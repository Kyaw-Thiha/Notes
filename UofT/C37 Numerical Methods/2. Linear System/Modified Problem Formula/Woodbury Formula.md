# Woodbury Formula 
#math #numerical-methods/linear-systems/gaussian-elimination/woodbury

The Woodbury formula generalizes Sherman–Morrison to handle rank-$k$ updates.  
For a matrix $A$ and matrices $U, V \in \mathbb{R}^{n \times k}$:
$$
(A + U V^T)^{-1} = A^{-1} + A^{-1} U ( I - V^T A^{-1} U )^{-1} V^T A^{-1}
$$
- **Rank-one case:** when $k = 1$, this reduces to the Sherman–Morrison formula.  
- **Cost:** about $O(n^2 k + k^3)$, much cheaper than recomputing a full $O(n^3)$ factorization.  
- **Use case:** efficiently update solutions when the matrix changes in a low-rank way.

⚠️ **Caution:** repeated updates may harm numerical stability, so in practice one may need to refactor occasionally.
## Practical method (avoiding explicit inversion)

If you already have an LU factorization of $A$, solving $(A + U V^T)x = b$ can be done by:
1. Solve \(AZ = U\)  → \(Z = A^{-1}U\)  
2. Solve \(Ay = b\)  → \(y = A^{-1}b\)  
3. Compute  
$$
x = y + Z \, (I - V^T Z)^{-1} V^T y
$$

This avoids forming $A^{-1}$ explicitly; only triangular solves, dot products, and a small $k \times k$ matrix inversion are needed.

## See Also
- [[Solving Linear Systems]]
- [[Gaussian Elimination]]
- [[Sherman-Morrison Formula]]