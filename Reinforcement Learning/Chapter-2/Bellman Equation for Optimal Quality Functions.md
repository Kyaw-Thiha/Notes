# Bellman Equation
#rl/bellman-equation/quality-function 
This note explore the relation between 
- optimal policy for [[Value Function|value function]] $\pi^{*}_{V}$
- and optimal policy for [[Quality Function|quality function]] $\pi^{*}_{V}$

Optimal state-value function is defined as
$$
\pi \geq \pi'
\iff
\forall s \in \mathcal{S}, \ V^{\pi}(s) \geq V^{\pi'}(s)
$$

Likewise, optimal action-value function is defined as
$$
\pi \geq \pi'
\iff
\forall \ (s,a) \in \mathcal{S} \times \mathcal{A}, 
\ Q^{\pi}(s, a) \geq Q^{\pi'}(s,a)
$$

---
**Proposition**: Given two policies $\pi, \ \pi'$, 
- $V^{\pi} \geq V^{\pi'} \implies Q^{\pi} \geq Q^{\pi^{'}}$ 
- and $V^{\pi} > V^{\pi'} \implies Q^{\pi} > Q^{\pi'}$.

**Proof**

Suppose that $V^{\pi}(s)$