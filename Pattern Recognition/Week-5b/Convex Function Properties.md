# Convex Function Properties
Here are additional properties of [[Convex Function|convex functions]] that are useful for machine learning.
You can find the definition and other properties of `convex functions` at [[Convex Function|here]].

---
## Composition of Convex & Linear Function

> The composition of `convex scalar function` and `linear function` is convex.

**Theorem**
Assume $f: \mathbb{R}^d \to \mathbb{R}$ is the composition $f(\mathbf{w}) = g(\langle \mathbf{w}, \mathbf{x} \rangle + y)$.
Where $\mathbf{x} \in \mathbb{R}^d$, $y \in \mathbb{R}$ and $g: \mathbb{R} \to \mathbb{R}$.
If $g$ is convex, then $f$ is convex too.

**Proof**
Let $\mathbf{w_{1}}, \mathbf{w}_{2} \in \mathbb{R}^d$ and $\alpha \in [0, 1]$.
Then,
$$
\begin{align}
&f(\alpha \mathbf{w}_{1}  
+ (1 - \alpha) \mathbf{w}_{2}) \\[6pt]

&= g( \ \langle \alpha \mathbf{w}_{1}  
+ (1 - \alpha)\mathbf{w_{2}}, \mathbf{x} \rangle + y 
\ ) \\[6pt]

&= g(\langle \alpha \mathbf{w}_{1}, \mathbf{x}  
\rangle + \langle (1 - \alpha)\mathbf{w}_{2}, 
 \mathbf{x} \rangle + y)  
& \text{by (1)}\\[6pt]

&= g( \ \alpha (\langle \mathbf{w}_{1}, \mathbf{x}  
\rangle + y) + (1 - \alpha)  
(\langle \mathbf{w}_{2}, \mathbf{x} \rangle + y) \ )
&\text{by (2)} \\[6pt]

&\leq \alpha \ g(\langle \mathbf{w}_{1}, \mathbf{x}  
\rangle + y) + (1 - \alpha) \ g(\langle \mathbf{w}_{2}, \mathbf{x} \rangle + y)
&\text{by (3)}
\end{align}
$$
where
- $(1)$: inner product distributes
- $(2)$: $y = \alpha y + (1 - \alpha) \ y$
- $(3)$: $g$ is convex

**Examples**
- $f(\mathbf{w}) = (\langle \mathbf{w}, \mathbf{x} \rangle - y)^2$
- $f(\mathbf{w}) = \log(1 + \exp(-y \langle \mathbf{w}, \mathbf{x} \rangle))$

---
## Maximum of Convex 
> Maximum of convex is convex

**Theorem**
For $i = 1, \dots, r$, let $f_{i}: \mathbb{R}^d \to \mathbb{R}$ be a [[Convex Function|convex function]].
Then, $g(\mathbf{x}) = \max_{i \in [r]} f_{i}(\mathbf{x})$ is convex.

**Proof**
$$
\begin{align}
&g(\alpha \mathbf{u} + (1 - \alpha) \mathbf{v})  
\\[6pt]
&= \max_{i} f_{i} (\alpha \mathbf{u} + (1 - \alpha) \mathbf{v}) \\[6pt]

&\leq \max_{i} (\alpha f_{i}(\mathbf{u})  
+ (1 - \alpha)f_{i}(\mathbf{v}) )  
&\text{by (1)} \\[6pt]

&\leq \alpha \max_{i} f_{i}(\mathbf{u})
+ (1 - \alpha) \max_{i} f_{i}(\mathbf{v})
&\text{by (2)} \\[6pt]

&= \alpha g(\mathbf{u}) + (1 - \alpha) g(\mathbf{v})
&\text{by (3)} \\[6pt]
\end{align}
$$
where
- $(1)$: $f_{i}$ is convex
- $(2)$: max of sum $\leq$ sum of max
- $(3)$: def'n of $g$

---
## Weighted Sum of Convex Function
For $i=1, \ \dots, \ r$, let $f_{i}: \mathbb{R}^d \to \mathbb{R}$ be a [[Convex Function|convex function]].
Then, $g(x) = \sum_{i \in [r]} w_{i} f_{i}(\mathbf{x})$ is `convex`.

$$
\begin{align}
&g(\alpha \mathbf{u} + (1 - \alpha)\mathbf{v})  
\\[6pt]

&= \sum_{i} w_{i} f_{i}(\alpha \ \mathbf{u} + (1 - \alpha) \ \mathbf{v}) \\[6pt]

&\leq \sum_{i} w_{i} ( \ \alpha f_{i}(\mathbf{u}) + (1-\alpha) f_{i}(\mathbf{v}) \ ) \\[6pt]

&= \alpha \sum_{i} w_{i} f_{i}(\mathbf{u}) 
+ (1 - \alpha) \sum_{i} w_{i} f_{i}(\mathbf{v})  
\\[6pt]

&= \alpha \ g(\mathbf{u})  
+ (1 - \alpha) \ g(\mathbf{v})
\end{align}
$$

`Jensen's inequality` follows from this result.
It states that for a random variable $X$ and a convex function $f$, $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$.

---
## See Also
- [[Convex Set]]
- [[Convex Function]]
