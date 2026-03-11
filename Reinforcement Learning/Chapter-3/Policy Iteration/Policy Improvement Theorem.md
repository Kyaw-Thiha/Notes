# Policy Improvement Theorem
#rl/planning/policy-iteration

> **Theorem**
> If for policies $\pi$ and $\pi'$ it holds that $T^{\pi'}Q^{\pi} = T^{*}Q^{\pi}$, we have that $Q^{\pi'} \geq Q^{\pi}$.

Essentially, [[Greedy Policy|greedy policy]] is a proper [[Policy Iteration|policy improvement]] step.

---
## Proof
**First step**: Show that $T^{\pi'}Q^{\pi} \geq Q^{\pi}$.
Note that $T^{\pi'}Q^{\pi} = T^{*}Q^{\pi}$ by assumption.
We also have $T^{*}Q^{\pi} \geq T^{\pi}Q^{\pi}$ as for any $(s,a) \in \mathcal{S} \times \mathcal{A}$, it holds that 
$$
\begin{align}
&r(s,a) + \gamma \int \mathcal{P}(ds' \mid s,a)
\max_{a' \in \mathcal{A}} Q^{\pi}(s', a') \\[6pt]
\geq \ &r(s,a) + \gamma \int \mathcal{P} 
(ds' \mid s,a) \ Q^{\pi}(s', \ \pi(s'))
\end{align}
$$
This shows that $T^{\pi'}Q^{\pi} = T^{*}Q^{\pi} \geq T^{\pi}Q^{\pi}$.
As $T^{\pi}Q^{\pi} = Q^{\pi}$, we can conclude that
$$
T^{\pi'}Q^{\pi} \geq Q^{\pi}
$$

**Second step**: Show that $Q^{\pi'} \geq Q^{\pi}$.
Applying $T^{\pi'}$ to both sides of $T^{\pi'}Q^{\pi} \geq Q^{\pi}$ and using the [[Monotonicity of Bellman Operator|monotonicity of the Bellman operator]], we get
$$
\begin{align}
T^{\pi'}(T^{\pi'} Q^{\pi})  
&\geq T^{\pi'} Q^{\pi} \\[6pt]
&= T^{*}Q^{\pi} \\[6pt]
&\geq Q^{\pi}
\end{align}
$$
Hence, we now have $(T^{\pi'})^{2}Q^{\pi} \geq Q^{\pi}$.
By repeating this argument for any $m \geq 1$, we get
$$
(T^{\pi'})^{m} Q^{\pi} \geq T^{*}Q^{\pi} \geq Q^{\pi}
$$
Taking the limit $m\to \infty$ and using the [[Contraction of Bellman Operator|contraction property of the Bellman operator]], we get
$$
\lim_{ m \to \infty } (T^{\pi'})^{m} \ Q^{\pi}
= Q^{\pi'}
$$
Combining the earlier inequality with the limit, we get
$$
Q^{\pi'}
= \lim_{ m \to \infty } (T^{\pi'})^{m} Q^{\pi}
\ \geq \ T^{*} Q^{\pi} 
\ \geq \ Q^{\pi}
$$
which is the desired result.

---
## Conclusion from the Proof
Choosing the [[Greedy Policy|greedy policy]] $w.r.t$ the most recent [[Value Function|value function]] gets us a new [[Markov Policy|policy]] that is at least as good as the previous one.

---
## See Also
- [[Policy Iteration]]
- [[Convergence of Policy Iteration Algorithm]]
- [[Greedy Policy]]
- [[Monotonicity of Bellman Operator]]
- [[Contraction of Bellman Operator]]
