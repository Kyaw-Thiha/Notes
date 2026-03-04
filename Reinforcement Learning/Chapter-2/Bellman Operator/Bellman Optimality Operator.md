# Bellman Optimality Operator
#rl/bellman-equation/operator 
The [[Bellman Operator|Bellman operators]] $T^{*}: \mathcal{B}(\mathcal{S}) \to \mathcal{B}(\mathcal{S})$ and $T^{*}: \mathcal{B} (\mathcal{S} \times \mathcal{A}) \to \mathcal{B}(\mathcal{S} \times \mathcal{A})$ are defined as mappings

$$
(T^{\pi}V)(s)
\triangleq \max \left\{  r^{\pi}(s) + \gamma \ \int 
\mathcal{P}(ds' \mid s,a) \ \pi(da \mid s)
\ V(s') \right\}
\ , \quad \forall s \in \mathcal{S}
$$
$$
(T^{\pi}Q)(s, a)
\triangleq r^{\pi}(s,a) + \gamma \ \int 
\mathcal{P}(ds' \mid s,a) \ \max_{a' \in \mathcal{A}}
Q(s', a') \ , \quad \forall(s,a) \in \mathcal{S} \times \mathcal{A}
$$

We can see that we can rewrite [[Bellman Equation for Optimal Value Functions|optimal value function]] $V^{*}$ and [[Bellman Equation for Optimal Quality Functions|optimal quality function]] $Q^{*}$ as
$$
\begin{align}
&V^{*} = T^{*} V^{*} \\[6pt]
&Q^{*} = T^{*} Q^{*}
\end{align}
$$

---
## Supremum of Bellman Operators
The maximization in Bellman optimality operator is defined over action space $\mathcal{A}$.
It can also be defined as maximization over space of policies.

Let the space of `stochastic policies` be defined as
$$
\Pi = \{ \pi: \pi(\cdot \mid s) \in \mathcal{M}(\mathcal{S}) \ , \quad \forall s \in \mathcal{S} \}
$$
Let the space of `deterministic policies` be defined as
$$
\Pi_{\det} = \{ \pi: \pi(s) \in \mathcal{A} 
\ , \quad \forall s \in \mathcal{S} \} 
= \mathcal{A}^{\mathcal{S}}
$$

Then for all $s \in \mathcal{S}$,
$$
\begin{align}
(T^{*}V)(s)
&= \sup_{a \in \mathcal{A}}  
\left\{  r(s,a) + \gamma \int  
\mathcal{P}(ds' \mid s, a) \ V(s') \right\} \\[6pt]

&= \sup_{\pi \in \Pi_{\det}}  
\left\{  r(s,a) + \gamma \int  
\mathcal{P}(ds' \mid s, a) \ V(s') \right\}  
&\quad (1) \\[6pt]

&= \sup_{\pi \in \Pi}  
\left\{  r(s,a) + \gamma \int  
\mathcal{P}(ds' \mid s, a) \ V(s') \right\}  
&\quad (2) \\[6pt]
\end{align}
$$

where
- $(1)$: $\Pi_{\det}$ is the Cartesian product of $\mathcal{A}$.
  So, maximizing action $a^{*}(s)$ over each dimensions $s \in \mathcal{S}$ can be combined to define $\pi^{*} = \prod_{s\in \mathcal{S}} a^{*}(s) \in \Pi_{\det}$.
- $(2)$: Longer proof

> This shows that the `Bellman optimality operator` is the `supremum` of the Bellman operator $T^{\pi}$ over all stochastic and deterministic policies.

For any policy $\pi \in \Pi$ and any function $V$, 
$$
T^{*}V \geq T^{\pi}V
$$

---
## See Also
- [[Bellman Operator]]
- [[Bellman Equation for Optimal Value Functions]]
- [[Bellman Equation for Optimal Quality Functions]]
