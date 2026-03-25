# Principle Component Analysis
#ml/models/pca 
Let $S = (\mathbf{x}_{1}, \ \dots, \ \mathbf{x}_{2})$ be vectors in $\mathbb{R}^{d}$.
We want
- a linear transformation that t

---
## Solving for PCA
**Lemma**: Let $(\mathbf{U}, \mathbf{W})$ be a solution to the PCA objective function.
Then, the columns of $\mathbf{U}$ are orthonormal $(i.e. \mathbf{U}^{T}\mathbf{U} = \mathbf{I})$ and $\mathbf{W} = \mathbf{U}^{T}$

**Proof**: Fix any $\mathbf{U}$,$\mathbf{W}$ and consider the mapping $x \mapsto \mathbf{U} \mathbf{W} \ x$.
The range (space of possible outputs) of this mapping is $R = \{ \mathbf{U}\mathbf{W}x: x \in \mathbb{R}^{d} \}$, and $n$ dimensional linear subspace of $\mathbb{R}^{d}$ (since $\mathbf{W}$ throws away all but $n$ dimensions)

Let $V \in \mathbb{R}^{d, \ n}$ be a matrix whose columns are an orthonormal basis of $\mathbb{R}$, and $\mathbf{V}\mathbf{V}^{T} = \mathbf{I} \in \mathbb{R}^{n, \ n}$.
