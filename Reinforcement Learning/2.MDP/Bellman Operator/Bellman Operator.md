# Bellman Operator
#rl/bellman-equation/operator
The `Bellman operators` are mapping from the space of value functions to the space of value functions.

---
## Definition
Given a policy $\pi: \mathcal{S} \to \mathcal{M}(\mathcal{A})$, the Bellman operator $T^{\pi}: \mathcal{B}(\mathcal{S}) \to \mathcal{B}(\mathcal{S})$ is defined as
$$
(T^{\pi}V)(s)
\triangleq r^{\pi}(s) + \gamma \ \int 
\mathcal{P}(ds' \mid s,a) \ \pi(da \mid s)
\ V(s') \ , \quad \forall s \in \mathcal{S}
$$
and the Bellman operator $T^{\pi}: \mathcal{B}(\mathcal{S} \times \mathcal{A}) \to \mathcal{B}(\mathcal{S} \times \mathcal{A})$ is defined as
$$
(T^{\pi}Q)(s, a)
\triangleq r^{\pi}(s,a) + \gamma \ \int 
\mathcal{P}(ds' \mid s,a) \ \pi(da' \mid s')
\ Q(s', a') 
$$

When the Bellman operator $T^{\pi}$ is applied to a function $V$, the result is another $T^{\pi}V$.

When we write $(T^{\pi}V)(s)$, we are evaluating $(T^{\pi}V)$ at the state $s \in \mathcal{S}$, so its value is a real number.

---
### Deterministic
If the policy is deterministic, the Bellman operators become
$$
(T^{\pi}V)(s)
\triangleq r^{\pi}(s) + \gamma \ \int 
\mathcal{P}(ds' \mid s, \ \pi(s)) 
\ V(s') \ , \quad \forall s \in \mathcal{S}
$$
and
$$
(T^{\pi}Q)(s, a)
\triangleq r^{\pi}(s,a) + \gamma \ \int 
\mathcal{P}(ds' \mid s,a) 
\ Q(s', \ \pi(s')) 
$$
---
### Compact form
We can write `Bellman operator` compactly as
$$
\begin{align}
&T^{\pi}V: V \mapsto r^{\pi} + \gamma \ \mathcal{P}^{\pi}V \\[6pt]
&T^{\pi}Q: Q \mapsto r + \gamma \ \mathcal{P}^{\pi} Q
\end{align}
$$

We can observe that Bellman equation for [[Bellman Equation for Value Function|value]]/[[Bellman Equation for Quality Function|quality function]] is the fixed point equation defined based on `Bellman Operator`:
$$
\begin{align}
&V^{\pi} = T^{\pi} V^{\pi} \\[6pt]
&Q^{\pi} = T^{\pi} Q^{\pi}
\end{align}
$$
which is a compact form of Bellman equations.

---
## See Also
- [[Bellman Equation for Value Function]]
- [[Bellman Equation for Optimal Quality Functions]]
- [[Bellman Optimality Operator]]
