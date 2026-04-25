# LU Factorization (Without Pivot)

If we want to solve $A\vec{x} = \vec{b}$, we can decompose $A$ into $L_{n-1},L_{n-2}, \dots, L_{1}A = U$ $\implies$ $A = \underbrace{L_{1}^{-1}, \dots, L_{n-1}^{-1}}_{L} \ U$
where
- $L$ is `Lower Triangular Matrix`
- $U$ is `Upper Triangular Matrix`

![LU Factorization](https://i.ytimg.com/vi/BFYFkn-eOQk/maxresdefault.jpg)

Hence, we get $LU\vec{x} = \vec{b}$
This allow us to solve the equation by solving $L\vec{d} = \vec{b}$ for $\vec{d}$ (`Forward Substitution`), and then solving $U\vec{x} = \vec{d}$ for $\vec{x}$ (`Backward Substitution`).

## Properties of Lower-Triangular $L$
- If $L_{i}$ is a `Gaussian Transformation`, then $L_{i}^{-1}$ exists and is also a `Gaussian Transformation`
- To compute $L_{i}^{-1}$, simply take $L_{i}$ and switch the signs of the multipliers (non-diagonal terms).
- If $L_{i}$ and $L_{j}$ are `Gaussian Transformations`, and $j>i$, then $L_{i}L_{j} = L_{i} + L_{j} - I$

## Examples
- [[LU Factorization Example-1|Example-1]]: This a simple example of `LU Factorization`
- [[LU Factorization Example-2|Example-2]]: This is an example of solving a linear equation using `LU Factorization`
