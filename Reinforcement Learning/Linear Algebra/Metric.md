# Metric
#math/linear-algebra/metric
A `metric` or a distance function on $\mathcal{Z}$ is a function $d: \mathcal{Z} \times \mathcal{Z} \to \mathcal{R}$ with the following properties:
- $d(x,y) \geq 0$ for all $x,y \in \mathcal{Z}$, and
  $d(x,y) = 0$ if and only if $x=y$
- **Symmetry**: $d(x,y) = d(y,x)$ for all $x,y \in \mathcal{Z}$
- **Triangle Inequality**: $d(x,y) \leq d(x,z) + d(z,y)$ for all $x,y,z \in \mathcal{Z}$

A `metric space` $(\mathcal{Z}, d)$ is a set $\mathcal{Z}$ equipped with a metric $d$.

**Example**
Let $\mathcal{Z} = \mathbb{R}$ and $d(x,y) = |x - y|$.
These together define a metric space $(\mathbb{R}, \ d)$.

---
## See Also
- [[Metric]]
