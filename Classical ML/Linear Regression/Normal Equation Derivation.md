# Normal Equation Derivation
#math 
Deriving $w^* = (\tilde{X}^\top \tilde{X})^{-1} \tilde{X}^\top y$ (Normal Equations)

We start from the (augmented) squared-error objective:
$$
E(\tilde{w}) \;=\; (y - \tilde{X}\tilde{w})^\top (y - \tilde{X}\tilde{w})
\;=\; \tilde{w}^\top \tilde{X}^\top \tilde{X}\tilde{w} \;-\; 2\,y^\top \tilde{X}\tilde{w} \;+\; y^\top y.
$$

We differentiate w.r.t. the vector $\tilde{w}$ and set the gradient to zero.

**Useful matrix-calculus rules** (for conformable dimensions, with $A$ constant):
- $\displaystyle \frac{\partial}{\partial w}\,(w^\top A w) = (A + A^\top)w$.
- If $A$ is symmetric, this becomes $2Aw$.
- $\displaystyle \frac{\partial}{\partial w}\,(c^\top w) = c$ and $\frac{\partial}{\partial w}\,(w^\top c) = c$.
- Constants (not depending on $w$) differentiate to $0$.

Apply these rules term by term:
1) Quadratic term:
$$
\frac{\partial}{\partial \tilde{w}}\big(\tilde{w}^\top \tilde{X}^\top \tilde{X}\tilde{w}\big)
= \big(\tilde{X}^\top \tilde{X} + (\tilde{X}^\top \tilde{X})^\top\big)\tilde{w}
= 2\,\tilde{X}^\top \tilde{X}\,\tilde{w}
\quad(\text{since } \tilde{X}^\top \tilde{X} \text{ is symmetric}).
$$

2) Linear term:
$$
\frac{\partial}{\partial \tilde{w}}\big(-2\,y^\top \tilde{X}\tilde{w}\big)
= -2\,\tilde{X}^\top y.
$$

3) Constant term:
$$
\frac{\partial}{\partial \tilde{w}}\big(y^\top y\big) = 0.
$$

Combine:
$$
\nabla_{\tilde{w}} E(\tilde{w}) \;=\; 2\,\tilde{X}^\top \tilde{X}\,\tilde{w} \;-\; 2\,\tilde{X}^\top y.
$$

Set the gradient to zero (first-order optimality condition):
$$
2\,\tilde{X}^\top \tilde{X}\,\tilde{w} - 2\,\tilde{X}^\top y = 0
\;\;\Longleftrightarrow\;\;
\tilde{X}^\top \tilde{X}\,\tilde{w} = \tilde{X}^\top y.
$$

These are the **normal equations**. If $\tilde{X}^\top \tilde{X}$ is invertible (i.e., $\tilde{X}$ has full column rank), solve:
$$
\tilde{w}^* \;=\; (\tilde{X}^\top \tilde{X})^{-1}\tilde{X}^\top y.
$$

**Notes & edge cases**
- If $\tilde{X}^\top \tilde{X}$ is singular (e.g., multicollinearity), use the Moore–Penrose pseudoinverse:
$$
\tilde{w}^* \;=\; \tilde{X}^+\,y.
$$
- With ridge (Tikhonov) regularization (penalty $\lambda \|\tilde{w}\|_2^2$), the solution becomes:
$$
\tilde{w}^* \;=\; (\tilde{X}^\top \tilde{X} + \lambda I)^{-1}\tilde{X}^\top y.
$$

(Here, augmentation means $\tilde{X} = [X\;\;\mathbf{1}]$ and $\tilde{w} = \begin{bmatrix}w \\ b\end{bmatrix}$, so the intercept is handled inside the same linear system.)
