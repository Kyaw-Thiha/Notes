# Norm
#math/linear-algebra/norm
A `norm` on a linear space $\mathcal{Z}$ is a function $||\cdot||: \mathcal{Z} \to \mathbb{R}$ with the following properties:
- **Non-negative**: For all $x \in \mathcal{Z}$, $||x|| \geq 0$.
- **Homogenous**: For all $x \in \mathcal{Z}$ and $\lambda \in \mathbb{R}$, $||\lambda x|| = ||\lambda|| \ ||x||$.
- **Triangle Inequality**: For all $x,y \in \mathcal{Z}$, $||x + y|| \leq ||x|| + ||y||$.
- **Strictly Positive**: For $x \in \mathcal{Z}$ we have that $||x|| = 0$ implies $x=0$.

> We can use a `norm` to define a distance between two points in a linear space $\mathcal{Z}$ by defining $d(x,y) = ||x - y||$.
> This gives us the [[Metric|metric space]] $(\mathcal{Z},  \ d)$.

---
## Examples
**Example-1**
Let $\mathcal{Z} = \mathbb{R}^{d}$ $(d \geq 1)$.
The following norms are used:
$$
\begin{align}
&||x||_{p} = \sqrt[p]{\sum ^{d}_{i=1} |x_{i}|^{p} }
 \quad , \quad 1 \leq p < \infty \\[6pt]
&||x||_{\infty} = \max_{i=1, \ \dots, \ d} |x_{i}|
\end{align}
$$

**Example-2**
Consider the space of continuous functions with domain $[0,1]$ denoted by $\mathcal{C}([0,1])$.
We can define the following norm for a function $f \in \mathcal{C}([0,1])$:
$$
||f||_{\infty} = \sup_{x \in [0, \ 1]}
| \ f(x) \ |
$$
This is called the `supremum` or `uniform norm`.
Given this norm, $( \ \mathcal{C}([0,1]), ||\cdot||_{\infty} \ )$ would be a normed linear space.

---
## See Also
- [[p-Norm]]
- [[1-Norm]]
- [[Infinity Norm]]
- [[Metric]]

---
