# Math Behind PCA
#ml/models/pca 

Here, we will be deriving the `PCA`.

![PCA|450](https://miro.medium.com/v2/1*qG4PEnoSWQRLoEL_P1ruhw.jpeg)

---
`Setting up the Problem`

Recall from [[Perturbed Reconstruction Error]] that $W^{\perp}\hat{X}^{\perp}$ is the `reconstruction error`.

Then, given a new point $\hat{x}^* \in R^{k \times 1}$, we can project back to reconstruct $\hat{y}^* \in R^{d \times 1}$.

$$
\hat{y}^* = W\hat{x}^* + \mu_{Y}
$$
Hence,
$$
\begin{cases}
y-\hat{y}^* = W^{\perp} \hat{x}^{*\perp} \\[6pt]
\hat{y}^* - \mu_{Y} = W\hat{x}
\end{cases}
$$

Since $(W\hat{x}^{*}) \perp (W^{\perp}\hat{x}^{*\perp})$, we can use `Pythagorean Theorem`
$$
||y - \mu_{Y}||^2 = ||y - \hat{y}^*||^2 + ||\hat{y}^* - \mu_{Y} ||^2
$$

---
`Reconstruction Error`

Applying `expectation`,
$$
\begin{align}
\text{Variance of data}  
&= \text{Reconstruction error} + \text{Variance of reconstruction} \\[6pt]

E[||y-\mu_{Y}||^2]  
&= E[||\hat{y}^* - y||^2] \ + \ E[||\hat{y}^* - \mu_{Y}||^2]
\end{align}
$$

This means that minimizing the `reconstruction error` will maximize the `variance of reconstruction`.

---
`Maximizing Variance`

Let the dataset be $Y \in R^{d \times N}$.
Then, the objective is to find a unit vector $w$ that maximizes 
$$
Var(w^TY) = w^TSw
$$

Note that since $n \cdot w \implies n^2 \cdot Var(w^TY)$, we need to constrain $w$ to be a unit vector
$$
\begin{align}
||w|| &= 1 \\[6pt]
w^Tw &= 1 \\[6pt]
w^Tw - 1 &= 0 \\[6pt]
g(w) &= 0 & \text{where } g(w)=w^Tw
\end{align}
$$

Hence, we can use [[Lagrange Multipliers]] to define the `Lagrangian` as
$$
L(w, \lambda) = w^TSw + \lambda(1 - w^Tw)
$$

---
`Solving the Lagrangian`

Maximizing $L(w, \lambda)$, we get
$$
\begin{align}
\frac{\partial L}{\partial w} &= 0 \\[6pt]
2Sw - 2\lambda w &= 0 \\[6pt]
Sw = \lambda w
\end{align}
$$

By definition,
- $\lambda$ is the `eigenvalue` of covariance matrix $S$
- $w$ is the `eigenvector` of covariance matrix $S$

---
`Relationship to Eigenvalue`

If you choose $w$ to be `eigenvector` of $S$, then the variance in that direction is equal to its `eigenvalue`.
$$
E(w) = w^TSw = w^T\lambda w = \lambda w^Tw = \lambda||w|| = \lambda
$$

This means that candidates for maximum `variance` are `eigenvectors`.

And to maximize `variance`, we need to pick the `eigenvector` whose `eigenvalue` is the largest.

---
`Properties of Eigenvectors & values`

Note that the `covariance matrix` $S$ is symmetric and positive semi definitive.
Hence,
- `Eigenvectors` are orthogonal to each other.
- `Eigenvalues` are real and non-negative
- Due to $||w||=1$, `eigenvectors` are orthonormal.

---
## See Also
- [[PCA Algorithm]]
- [[Lagrange Multipliers]]
- [[Perturbed Reconstruction Error]]
- [[PCA Example]]

