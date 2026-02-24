# Lipschitzness
Let $C \subset \mathbb{R}^d$.
A function $f: \mathbb{R}^d \to \mathbb{R}^k$ is $\rho \text{-Lipschitz}$ over $C$ if for every $\mathbf{w}_{1}, \mathbf{w_{2}} \in C$,

$$
||f(\mathbf{w}_{1}) - f(\mathbf{w}_{2})||_{2}
\ \leq \ 
\rho \ ||\mathbf{w}_{1} - \mathbf{w}_{2} ||_{2}
$$

![Lipschitzness|300](https://upload.wikimedia.org/wikipedia/commons/5/58/Lipschitz_Visualisierung.gif)

In other words, a `Lipschitz function` cannot change 'too quickly' $(\text{i.e: it's gradient is bounded})$.

---
**Formal Definition**
Consider differentiable scalar function $f: \mathbb{R} \to \mathbb{R}$.
By the `mean value theorem`,
$$
f(w_{1}) - f(w_{2})
= f'(u) (w_{1} - w_{2})
$$
where $u \in ] w_{1}, w_{2} [$.
Hence if $f'$ is bounded everywhere by $\rho$, then $f$ is $\rho \text{-Lipschitz}$.

---
## Composition preserves Lipschitzness

**Theorem**
Let $f(\mathbf{x}) = g_{1}(g_{2}(\mathbf{x}))$, where $g_{1}$ is $\rho_{1}\text{-Lipschitz}$ and $g_{2}$ is $\rho_{2}\text{-Lipschitz}$.
Then, $f$ is $(\rho_{1} \rho_{2})\text{-Lipschitz}$.

In particular, if $g_{2}$ is the `linear function` $g_{2}(\mathbf{x}) = \langle \mathbf{v}, \mathbf{x} \rangle + b$ for some $\mathbf{v} \in \mathbb{R}^d$, $b \in \mathbf{R}$, then $f$ is $(p_{1}\ ||\mathbf{v}||)\text{-Lipschitz}$.

**Proof**
$$
\begin{align}
&|f(\mathbf{w_{1}}) - f(\mathbf{w}_{2})| \\[6pt]

&= | \ g_{1} (g_{2}(\mathbf{w}_{1}))  
- g_{1}(g_{2}(\mathbf{w}_{2})) \ | \\[6pt]

&\leq \rho_{1} \ ||g_{2}(\mathbf{w}_{1})  
- g_{2}(\mathbf{w}_{2})|| \\[6pt]

&\leq \rho_{1} \ \rho_{2} \ || \ \mathbf{w}_{1} - \mathbf{w}_{2} \ ||
\end{align}
$$

---

