# Sherman–Morrison Formula 
#math #numerical-methods/linear-systems/gaussian-elimination/sherman-morrison

For a rank-one update of a matrix $A$ with vectors $u, v$, the inverse can be expressed as:

$$
(A + uv^T)^{-1} = A^{-1} + \frac{A^{-1} u v^T A^{-1}}{1 - v^T A^{-1} u}
$$

This reduces the cost from $O(n^3)$ (full inversion) to $O(n^2)$ (matrix–vector operations).  
However, explicit inversion is usually avoided due to numerical inaccuracy.

## Practical method (using LU factorization)

To solve $(A + uv^T)x = b$ without forming an inverse:
1. Solve $Az = u$  → $z = A^{-1}u$  
2. Solve $Ay = b$  → $y = A^{-1}b$  
3. Compute  
$$
x = y + \frac{v^T y}{1 - v^T z} \, z
$$

This approach only requires triangular solves and dot products, costing $O(n^2)$.

## See Also
- [[Solving Linear Systems]]
- [[Gaussian Elimination]]
- [[Woodbury Formula]]
