# Convex-Lipschitz-Bounded Learning Problem

A [[Convex Learning Problems|learning problem]] $(\mathcal{H}, \mathcal{Z}, \mathcal{l})$ is called `Convex-Lipshitz-Bounded`, with parameters $\rho, \ B$ if the following holds:
- $\mathcal{H}$ is a [[Convex Set|convex set]]  
- $\forall \mathbf{w} \in \mathcal{H}, \ ||\mathbf{w}|| \leq B$ where $B$ is the bound
- $\forall z \in \mathcal{Z}, \ l(\cdot, z)$ is a [[Convex Function|convex function]] and $\rho$[[Lipschitzness|-Lipschitz function]]

---
### Example
Let 
- $\mathcal{X} = \{ x \in \mathbb{R}^d: ||x|| \leq \rho \}$ and $\mathcal{Y} = \mathbb{R}$.
- $\mathcal{H} = \{ \mathbf{w} \in \mathbb{R}^d : ||\mathbf{w}|| \leq B \}$ be the [[Convex Set|convex set]].
- $l(\mathbf{w}, (\mathbf{x}, \mathbf{y})) = |\langle \mathbf{w}, \ \mathbf{x} - \mathbf{y} \rangle|$ be the [[Loss Function|loss function]] for regression with absolute value loss.

Because the instances $\mathbf{x}$ are in $\rho \text{-ball}$, 
$\langle \mathbf{w}, \mathbf{x} \rangle$ is $\rho \text{-Lipschitz}$ and $|\cdot|$ is $1\text{-Lipschitz}$.
So, $l$ is $\rho \text{-Lipschitz}$ and convex.
Because $\mathbf{w}$ lie in the $\text{B-ball}$, the hypothesis class is bounded and convex.

Therefore, [[Linear Regression|linear regression]] with absolute value loss is `Convex-Lipschitz-Bounded` with parameters $\rho, \ B$.

---
## See Also
- [[Convex Learning Problems]]
- [[Convex Set]]
- [[Convex Function]]

