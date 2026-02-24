# Ridge Regression
`Ridge Regression` (Tikhonov Regularization) can be denoted as
$$
\arg \min_{\mathbf{w} \in \mathbb{R}^d}
\left( \frac{1}{m} \sum^m_{i=1} \frac{1}{2}
(\mathbf{w} \cdot \mathbf{x}_{i} - y_{i})^2 + \lambda||\mathbf{w}||^2_{2} \right)
$$
To solve this, we compute the gradient $w.r.t$ $\mathbf{w}$, set it to zero and solve:
$$
(2\lambda m \ \mathbf{I} + \mathbf{A})\mathbf{w}
= \mathbf{b}
$$

using standard methods where
$$
\mathbf{A}
= \left( \sum^m_{i=1} \mathbf{x}_{i} 
\mathbf{x}_{i}^T \right)
\quad \text{and} \quad
\mathbf{b} = \sum^m_{i=1} y_{i} \mathbf{x}_{i}
$$

Note that $\mathbf{A}$ is a `positive semidefinite matrix` (non-negative eigenvalues), and may not be invertible (zero-valued eigenvalues).

Adding $2\lambda m \ \mathbf{I}$ will ensure the eigenvalues are lower-bounded by $2\lambda m$ and the matrix will be invertible.

---
## Ridge Regression is Learnable on Average

**Claim**
Let $\mathcal{D}$ be a distribution over $\mathcal{X} \times [-1, +1]$, 
where $\mathcal{X} = \{ \mathbf{x} \in \mathbb{R}^d: ||\mathbf{x}|| \leq 1 \}$
For any $\epsilon \in ]0, 1[$, let $m \geq 150B^2/\epsilon^2$.

Then, applying `ridge regression` algorithm with parameters $\lambda=\frac{\epsilon}{3B^2}$ satisfies

$$
\mathbb{E}_{S \sim \mathcal{D}^m}
[ \ L_{\mathcal{D}}(A(S)) \ ]
\leq \min_{\mathbf{w} \in \mathcal{H}}
L_{\mathcal{D}}(\mathbf{w}) + \epsilon
$$
Note that this differs from the standard definition of [[Agnostic PAC Learning]], in that 
- it bounds the `expected risk` of the learner (average over the datasets),
- as opposed to the `risk of the individual` learner being bounded with at least probability $1 - \delta$.

---
## See Also
- [[Regularization]]
- [[Regularized Loss Minimization (RLM)]]
- [[Linear Regression]]