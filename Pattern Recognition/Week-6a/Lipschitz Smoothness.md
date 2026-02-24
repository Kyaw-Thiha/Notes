# Smoothness 
### Lipschitz-continuous gradients
A differentiable function $f:\mathbb{R}^d \to \mathbb{R}$ is $\beta \text{-smooth}$ if its gradient is $\beta$-[[Lipschitzness|Lipschitz]].
Namely, for all $\mathbf{v}, \mathbf{w}$, we have
$$
|\nabla f(\mathbf{v}) - \nabla f(\mathbf{w})|
\leq \beta ||\mathbf{v} - \mathbf{w}||
$$

---
### Bounding using Smoothness
It is possible to show that `smoothness` implies that $\forall \mathbf{v}, \mathbf{w}$,
$$
f(\mathbf{v})
\leq f(\mathbf{w}) + \langle \nabla f(\mathbf{w})
, \mathbf{v} - \mathbf{w} \rangle
+ \frac{\beta}{2} ||\mathbf{v} - \mathbf{w}||^2
$$

Recall that [[Convex Function|convex function]] $f$ means that
$$
f(\mathbf{v}) \geq f(\mathbf{w})
+ \langle \nabla f(\mathbf{w}), \mathbf{v} 
- \mathbf{w} \rangle
$$

If it is a smooth convex function, we can upper and lower bound the error in approximating the function with a gradient.
Set $\mathbf{v} = \mathbf{w} - \frac{1}{\beta} \nabla  f(\mathbf{w})$.
Substituting it in the equation above, we get
$$
\frac{1}{2\beta}
|| \nabla f(\mathbf{w}) ||^2
\leq f(\mathbf{w}) - f(\mathbf{v})
$$
If we assume that $\forall \mathbf{v}, f(\mathbf{v}) \geq 0$, then `smoothness` implies $f$ is a `self-bounded function`:
$$
|| \ \nabla f(\mathbf{w}) \ ||^2
\leq 2 \beta \ f(\mathbf{w})
$$

---
## Smoothness Composes

**Theorem**
Let $f(\mathbf{w}) = g(\langle \mathbf{w}, \mathbf{v} \rangle + b)$ where 
- $g: \mathbb{R} \to \mathbb{R}$ is a $\beta \text{-smooth}$ function
- $x \in \mathbb{R}^d$
- $b \in \mathbb{R}$

Then, $f$ is $(\beta \ ||\mathbf{x}||^2)$-smooth.

**Proof**
$$
\begin{align}
f(\mathbf{w}) 
&= g(\langle \mathbf{w}, \mathbf{x} \rangle + b)
\\[6pt]

&\leq g(\langle \mathbf{w}, \mathbf{x} \rangle + b)
+ g'(\langle \mathbf{w}, \mathbf{x} \rangle + b)
\langle \mathbf{v} - \mathbf{w}, \mathbf{x} \rangle
+ \frac{\beta}{2} (\langle \mathbf{v}  
- \mathbf{w}, \mathbf{x} \rangle)^2
&\text{by (1)} \\[6pt]

&\leq g(\langle \mathbf{w}, \mathbf{x} \rangle + b)
+ \langle \mathbf{v} -\mathbf{w},  
g'(\langle \mathbf{w},\mathbf{x}\rangle + b)  
\ \mathbf{x} \rangle
+ \frac{\beta}{2} (|| \mathbf{v}  
- \mathbf{w}|| \ ||\mathbf{x}|| )^2
&\text{by (2)} \\[6pt]

&\leq g(\langle \mathbf{w}, \mathbf{x} \rangle + b)
+ \langle \mathbf{v} -\mathbf{w},  
\nabla f(\mathbf{w}) \rangle
+ \frac{\beta \ ||\mathbf{x}||^2}{2}  
\ || \mathbf{v} - \mathbf{w}||^2
&\text{by (3)} \\[6pt]
\end{align}
$$
where
- $(1)$: $g$ is smooth
- $(2)$: C-S inequality
- $(3)$: The chain rule

---
## See Also
- [[Lipschitzness]]
