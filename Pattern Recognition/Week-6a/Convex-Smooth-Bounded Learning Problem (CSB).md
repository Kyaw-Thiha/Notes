# Convex-Smooth-Bounded Learning Problem
A [[Convex Learning Problems|learning problem]] $(\mathcal{H}, \mathcal{Z}, l)$ is a `Convex-Smooth-Bounded` with parameters $\beta, \ B$ if the following holds
- $\mathcal{H}$ is a [[Convex Set|convex set]] and $\forall \mathbf{w} \in \mathcal{H}, \ ||\mathbf{w}|| \leq B$.
- $\forall \mathbf{z} \in \mathcal{Z}, \ l(\cdot, \mathbf{z})$ is a [[Convex Function|convex]], non-negative and $\beta$[[Lipschitz Smoothness|-smooth]] function.

Note that the `non-negativity` of $l$ is to ensure [[Loss Function|loss]] is `self-bounded`.

---
## Example
Let 
- $\mathcal{X}=\left\{  \mathbf{x} \in \mathbb{R}^d: ||\mathbf{x}|| \leq \frac{\beta}{2}  \right\}$ and $\mathcal{Y} = \mathbb{R}$  be the `input/output space`.
- $\mathcal{H} = \{ \mathbf{w} \in \mathbb{R}^d: || \mathbf{w} || \leq B \}$ be the `hypothesis class`.
- $l(\mathbf{w}, (\mathbf{x}, \ \mathbf{y})) = (\langle \mathbf{w}, \mathbf{x} \rangle - \mathbf{y})^2$ is the [[Loss Function|loss function]] for [[Linear Regression|least squared regression]].

Note that $x^2$ is $2\text{-smooth}$. 
And the instances $\mathbf{x}$ comes from the $\frac{\beta}{2}\text{-ball}$.
Hence, the `least squares function` is $\beta \text{-smooth}$.

Since $\mathbf{w}$ is bounded by the $\text{B-ball}$, $\mathcal{H}$ is bounded and convex.

Therefore, [[Linear Regression|linear regression]] with absolute loss is `Convex-Lipschitz-Bounded` with parameters $\rho, \ B$.

---
