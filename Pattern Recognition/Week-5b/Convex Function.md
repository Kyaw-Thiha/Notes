# Convex Function

> `Defn` 
> Let $C$ be a convex set.
> A function $f: C \to \mathbb{R}$ is convex if,
> for every $\mathbf{u}, \mathbf{v} \in C$  and $\alpha \in [0, 1]$, then 
$$
f(\alpha \mathbf{u} + (1 - \alpha)\mathbf{v})
\leq \alpha f(\mathbf{u}) + (1 - \alpha) f(\mathbf{v})
$$

---
## Epigraph
> The `epigraph` of a convex function is a convex set.

![Epigraph](https://i.sstatic.net/BQbZe.png)

The epigraph of a function $f$ is the set
$$
\text{epigraph}(f)
= \{ (x, \beta): f(x) \leq \beta \}
$$
where $\beta \in \mathbb{R}$.

> A function is convex if and only if its epigraph is a convex set.

---
## Global Minima
> `Convex Functions` have one global minima.

The reason we want functions to be convex is because every local minimum of the function is also a global minimum.

**Math Proof**
Let $B(\mathbf{u}, \mathbf{r}) = \{ \mathbf{v}: || \mathbf{v} - \mathbf{u} || \leq r \}$.
Then, $f(\mathbf{u})$ is a `local minimum` of $f$ at $\mathbf{u}$ if there exists some $r > 0$ such that $\forall \mathbf{v} \in B(\mathbf{u}, r)$, then $f(\mathbf{v}) \geq f(\mathbf{u})$.

It follows that for any $\mathbf{v}$ (but not $B$ because local minima), $\exists \alpha > 0$ small enough that $\mathbf{u} + \alpha(\mathbf{v} - \mathbf{u}) \in B(\mathbf{u}, r)$ and therefore
$$
f(\mathbf{u}) 
\leq f(\mathbf{u} + \alpha(\mathbf{v} -\mathbf{u}))
$$
and if $f$ is convex,
$$
\begin{align}
f(\mathbf{u} + \alpha(\mathbf{v} - \mathbf{u}))  
&= f((1 - \alpha) \mathbf{u} + \alpha \mathbf{v}) \\[6pt]

f((1 - \alpha) \mathbf{u} + \alpha \mathbf{v})
&\leq (1 - \alpha) f(\mathbf{u})  
+ \alpha f(\mathbf{v}) \\[6pt]

\implies f(\mathbf{u})
&\leq (1 - \alpha) f(\mathbf{u})  
+ \alpha f(\mathbf{v}) \\[6pt]

\implies f(\mathbf{u}) &\leq f(\mathbf{v})
\end{align}
$$

---
## Tangents Lie Below Convex Functions
For any $\mathbf{w}$, we can construct a tangent to $f$ at $\mathbf{w}$ that lies below $f$ everywhere.
If $f$ is differentiable, we can define the tangent as
$$
l(\mathbf{u})
= f(\mathbf{w}) + \langle \ \nabla_{\mathbf{w}} 
f(\mathbf{w}), \mathbf{u} - \mathbf{w} \ \rangle
$$
where
- $\nabla_{\mathbf{w}}f(\mathbf{w}) = \left( \frac{\partial f(\mathbf{w})}{\partial w_{1}}, \dots, \frac{\partial f(\mathbf{w})}{\partial w_{d}} \right)$

So for convex differentiable functions, we get
$$
\forall \mathbf{u},
f(\mathbf{u}) \geq f(\mathbf{w}) + 
\langle \nabla_{\mathbf{w}} f(\mathbf{w}), 
\mathbf{u} - \mathbf{w} \rangle
$$

---
## Determining Convexity
Let $f:\mathbb{R} \to \mathbb{R}$ be a scalar, twice differentiable function.
Let $f'$ and $f''$ be its first and second derivatives respectively.

Then, the following properties are equivalent:
1. $f$ is `convex`
2. $f'$ is `monotonically non-decreasing`

$$
x_{1} < x_{2} 
\implies f'(x_{1}) \leq f'(x_{2})
$$
3. $f''$ is `nonnegative`

$$
f''(x) \geq 0, \forall x \in \mathbb{R}
$$

---
## See Also
- [[Convex Set]]
- [[Convex Function Properties]]
