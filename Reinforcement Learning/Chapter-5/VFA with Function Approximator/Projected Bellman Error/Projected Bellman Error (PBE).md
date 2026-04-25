# Projected Bellman Error

> **Main Idea**:
> The distance between a [[Value Function|value function]] $V \in \mathcal{F}$ and the projection of $T^{\pi}V$ onto $\mathcal{F}$ should be made small.

![Projected Bellman Error|300](https://notes-media.kthiha.com/Projected-Bellman-Error/57dffaf60c60af564795232dc8f69878.png)

We find a $V \in \mathcal{F}$ such that
$$
V = \Pi_{\mathcal{F}, \mu} \ T^{\pi}V
$$
where $\Pi_{\mathcal{F}, \mu}$ is the `projection operator` onto $\mathcal{F}$.

---
## Projection Operator
The projection operator $\Pi_{\mathcal{F}, \mu}$ is a linear operator that 
- takes $V \in \mathcal{B(S)}$ and maps it to the closest point on $\mathcal{F}$,
- measured according to its $L_{2}(\mu)$ [[Norm|norm]].

$$
\Pi_{\mathcal{F}, \mu}V
\triangleq \arg\min_{V' \in \mathcal{F}}
||V' - V||_{2, \mu}
$$
If the choice of distribution $\mu$ is clear from the context, we may omit it.

Some properties of this projection operator includes:
- $\Pi_{\mathcal{F}, \mu}V \in \mathcal{F}$
- If $V \in \mathcal{F}$, we have $\Pi_{\mathcal{F}, \mu}V = V$.
- The projection operator onto a subspace (also a [[Convex Set|closed convex set]]) is a non-expansion. 

$$
||\Pi_{\mathcal{F}, \mu}V_{1} 
- \Pi_{\mathcal{F}, \mu}V_{2}||_{2, \mu}
\leq ||V_{1} - V_{2}||_{2, \mu}
$$

---
## Defining the Loss Function
We can define a [[Loss Function|loss function]] based on $V=\Pi_{\mathcal{F}} \ T^{\pi}V$.
We can use different [[Norm|norms]].
A common choice is the $L_{2}(\mu)$-[[Norm|norm]]:
$$
||V - \Pi_{\mathcal{F}} \ T^{\pi}V||_{2, \mu}
$$
This is called `Projected Bellman Error` or `Mean Squared Projected Bellman Error` (`MSPBE`).

We find the [[Value Function|value function]] by solving the following optimization problem:
$$
V \leftarrow \arg\min_{V \in \mathcal{F}}
||V - \Pi_{\mathcal{F}} \ T^{\pi}V||_{2, \mu}
$$

As $V \in \mathcal{F}$, 
$$
\begin{align}
V - \Pi_{\mathcal{F}, \mu} \ T^{\pi}V

&= \Pi_{\mathcal{F}, \mu} V  
- \Pi_{\mathcal{F}, \mu} T^{\pi}V \\[6pt]

&= \Pi_{\mathcal{F}, \mu} (V - T^{\pi}V) \\[6pt]

&= -\Pi_{\mathcal{F}, \mu}(\text{BR}(V))
\end{align}
$$
where $\text{BR(V)}$ is the [[Bellman Residual Minimization (BRM)|Bellman residual]].

So, the loss is 
$$
||V - \Pi_{\mathcal{F}} \ T^{\pi}V||_{2, \mu}
= || \ \Pi_{\mathcal{F}, \mu}(\text{BR}(V)) 
\ ||_{2, \mu}
$$
the norm of the projection of the [[Bellman Residual Minimization (BRM)|Bellman residual]] onto $\mathcal{F}$.

---
## Comparism to [[Bellman Residual Minimization (BRM)|BRM]]
![Projected Bellman Error|250](https://notes-media.kthiha.com/Projected-Bellman-Error/57dffaf60c60af564795232dc8f69878.png)
**Bellman Residual Minimization**:
$$
V \leftarrow \arg\min_{V \in \mathcal{F}}
||V - T^{\pi}V||_{p, \mu}
= || \ \text{BR}(V) \ ||_{2, \mu}
$$

**Projected Bellman Error**:
$$
V \leftarrow \arg\min_{V \in \mathcal{F}}
||V - \Pi_{\mathcal{F}} \ T^{\pi}V||_{2, \mu}
= || \ \Pi_{\mathcal{F}, \mu}(\text{BR}(V)) \ ||_{2, \mu}
$$

---
## Nested Formulation of Projected Bellman Error
We can think of [[Projected Bellman Error (PBE)|PBE]] as simultaneously solving these two coupled (nested) optimization problems:
$$
\begin{align}
&V \leftarrow \arg\min_{V' \in \mathcal{F}}
||V' - \tilde{V}(V')||^{2}_{2, \mu} \\[6pt]

&\tilde{V}(V') \leftarrow \arg\min_{V'' \in 
 \mathcal{F}} ||V'' - T^{\pi}V'||^{2}_{2, \mu}
\end{align}
$$

- If $\mathcal{F}$ is a linear function space, the projection has a closed-form solution.
- For more general spaces, the solution may not be simple
- Regularized variants: suitable for avoiding overfitting when $\mathcal{F}$ is a very large function space:

$$
\begin{align}
&V \leftarrow \arg\min_{V' \in \mathcal{F}}
||V' - \tilde{V}(V')||^{2}_{2, \mu} 
+ \lambda J(V') \\[6pt]

&\tilde{V}(V') \leftarrow \arg\min_{V'' \in 
 \mathcal{F}} ||V'' - T^{\pi}V'||^{2}_{2, \mu}
  + \lambda J(V'')
\end{align}
$$

---
## Solving PBE
