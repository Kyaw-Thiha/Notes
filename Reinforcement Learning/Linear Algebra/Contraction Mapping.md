# Contraction Mapping
#math/linear-algebra/contraction-mapping
Let $(\mathcal{Z}, \ d)$ be a [[Metric|metric space]].
A mapping $L: \mathcal{Z} \to \mathcal{Z}$ is a `contraction mapping` if there exists a constant $0 \leq a < 1$ such that for all $z_{1}, z_{2} \in \mathcal{Z}$, we have
$$
d(L(z_{1}), \ L(z_{2}))
\leq a \ d(z_{1}, \ z_{2})
$$

![Contraction Mapping|500](https://notes-media.kthiha.com/Contraction-Mapping/9df5a42f846fd5eceb87a53e0418c1ea.png)

---
## Example
Let $\mathcal{Z}=\mathbb{R}$ and $d(z_{1}, \ z_{2}) = |z_{1} - z_{2}|$.
Consider the mapping $L : z \mapsto az$ for $a \in \mathbb{R}$.
For any $z_{1}, z_{2} \in \mathbb{R}$, we have
$$
\begin{align}
d( \ L(z_{1}), \ L(z_{2}) \ )
&= |L(z_{1}) - L(z_{2})| \\[6pt]
&= |az_{1} - az_{2}| \\[6pt]
&= |a| \ |z_{1} - z_{2}| \\[6pt]
&= |a| \ d(z_{1}, \ z_{2})
\end{align}
$$
So if $|a| < 1$, this is a `contraction mapping`.

---
## Utility of Contraction Mapping
- Describes the `stability` behaviour of a dynamic system.
  Stability is related to the uniqueness of where the dynamical system converges.
- Can be used to show `uniqueness` of solution.
- Can sometimes be used when solving equations.

---
### Example on relation to stability
As an example of its relation to stability, 
let $z_{0} \in \mathcal{Z}$ and consider a mapping $L: z \mapsto az$ for some $a \in \mathbb{R}$.

Define the dynamical system
$$
z_{k+1} = Lz_{k} \ , \quad k = 0, \ 1, \ \dots
$$
The dynamical system described by this mapping generates
$$
\begin{align}
&z_{0} \\[6pt]
&z_{1} = az_{0} \\[6pt]
&z_{2} = az_{1} = a^{2} \ z_{0} \\[6pt]
& \quad \ \vdots \\[6pt]
&z_{k} = az_{k-1} = a^{k} \ z_{0}
\end{align}
$$
Hence, its general formula is
$$
z_{k} = a^{k} \ z_{0}
$$
- If $|a| < 1$, $z_{k}$ converges to zero no matter what $z_{0}$ is.
- If $a=1$, we have $z_{k} = z_{0}$.
  So, depending on $z_{0}$, it converges to different points.
- For $a = -1$, the sequence would oscillate between $+z_{0}$ and $-z_{0}$.
- If $|a| > 1$, the sequence diverges (unless $z_{0} = 0$)

---
## See Also
- [[Metric]]
- [[Norm]]
- [[Fixed Point]]