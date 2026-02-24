# Subgradient

**Lemma**
Let $C$ be an [[Convex Set|open convex set]].
$f: C \to \mathbb{R}$ is convex if and only if $\forall \mathbf{w} \in C, \exists \mathbf{v}$ such that
$$
\forall \mathbf{u} \in C, 
f(\mathbf{u}) \geq f(\mathbf{w}) 
+ \langle \mathbf{u} - \mathbf{w}, \mathbf{v} \rangle
$$
> For any [[Convex Function|convex function]], there is a tangent plane that is below the function everywhere.

![Subgradient|400](https://www.researchgate.net/profile/Alex-Olshevsky/publication/320055319/figure/fig1/AS:644645650456579@1530706908806/An-illustration-of-the-definition-of-a-subgradient-At-the-point-x0-the-function-shown.png)

**Subgradient Definition**
- `Subgradient` of $f$ at $\mathbf{w}$ is a vector $\mathbf{v}$ that satisfies the relationship above.
- `Differential set` $\partial f(\mathbf{w})$ is the set of subgradients of $f$ at $\mathbf{w}$

---
## Constructing Subgradients for Pointwise Maximum Functions
**Claim**
> Let $g(\mathbf{w}) = \max_{i \in [r]} \ g_{i}(\mathbf{w})$ 
> for $r$ [[Convex Function|convex]], differentiable functions $g_{1}, \ \dots, \ g_{r}$. 
> Given some $\mathbf{w}$, let $j \in \arg \max_{i} g_{i}(\mathbf{w})$.
> Then, $\nabla g_{j}(\mathbf{w}) \in \partial g(\mathbf{w})$.

**Proof**
Since $g_{j}$ is [[Convex Function|convex]], then $\forall \mathbf{u}$:
$$
g_{j}(\mathbf{u}) 
\geq g_{j}(\mathbf{w}) 
+ \langle \mathbf{u} - \mathbf{w}, 
\ \nabla g_{j}(\mathbf{w}) \rangle
$$
Since $g(\mathbf{w}) = g_{j}(\mathbf{w})$ and $g(\mathbf{u}) \geq g_{j}(\mathbf{u})$, it follows that
$$
g(\mathbf{u}) \geq g(\mathbf{w}) 
+ \langle \mathbf{u} - \mathbf{w}, 
\ \nabla g_{j}(\mathbf{w}) \rangle
$$

---
## Examples
### The Absolute Function
Consider $f(x) = |x|$.
We can construct the differential set over the entire domain.

Since $|x|$ is only non-differentiable at $x=0$, that is where the only non-trivial differential set lives:

$$
\partial f(x) =
\begin{cases}
\{ -1 \} & \text{if } x<0 \\[6pt]
\{ -1, 1 \} & \text{if } x=0 \\[6pt]
\{ 1 \} & \text{if } x>0 \\[6pt]
\end{cases}
$$

We have now defined the `subgradient` as being any value in the range $[-1, \ 1]$ but practically speaking we really only need `one subgradient` to do optimization.

---
### Subgradient for Hinge Loss
We defined `Hinge Loss` to be $\ell^{\text{hinge}}(\mathbf{w}) = \max \{ 0, \ 1-y(\mathbf{w} \cdot \mathbf{x}) \}$.
To compute a subgradient $\mathbf{v}$ at $\mathbf{w}$, we rely on the [[#Constructing Subgradients for Pointwise Maximum Functions|preceding claim]] about pointwise maximum functions to get:
$$
\mathbf{v} =
\begin{cases}
0 & \text{if } 1-y(\mathbf{w} \cdot \mathbf{x})  
\leq 0 \\[6pt]

-y \ \mathbf{x} & \text{if } 1-y(\mathbf{w} \cdot  
\mathbf{x}) > 0
\end{cases}
$$
The [[Gradient Descent Detail|gradient descent algorithm]] can then be modified to use `subgradients` of $f(\mathbf{w})$ at $\mathbf{w}^{(t)}$.

---
## See Also
- [[Gradient Descent]]
- [[Gradient Descent Review]]
- [[Gradient Descent Detail]]

