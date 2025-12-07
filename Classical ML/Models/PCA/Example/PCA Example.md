# PCA Example

`Perfect Reconstruction`

![[PCA (Centering).png|300]]

`Dimension Reduction`

When we carry out [[Centering Coordinate System]] and [[Rotating Coordinate System]], we can notice that the `covariance matrix` still denotes the same total variance.

However, the `covariance matrix` achieved through [[Finding the Orthogonal Matrix]] concentrate the `total variance` to a single dimension.
Hence, we can use it to carry out [[Dimensionality Reduction]] by removing the other dimension.

---
`Reconstruction`

When we try to reconstruct the original dataset from this `dataset after dimension reduction`, we get a [[Reconstruction Error]] of zero.

However, perturbing the dataset and computing the [[Perturbed Reconstruction Error]], we can notice that $W^{\perp}\hat{X}^{\perp}$ is the `reconstruction error`.

---
## See Also
- [[Centering Coordinate System]]
- [[Rotating Coordinate System]]
- [[Finding the Orthogonal Matrix]]
- [[Eigenstructure of Symmetric Matrices]]
- [[Dimension Reduction]]
- [[Reconstruction Error]]
- [[Perturbed Reconstruction Error]]
