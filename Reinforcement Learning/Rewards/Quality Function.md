# Quality Function

`Q-Function` defines the expected cumulative future [[Reward|reward]] of taking a specific action in a given state.

![Quality Function](https://cugtyt.github.io/blog/rl-notes/R/q-func-eq.png)

---
## Definition
Let $(R_{t}; \ t\geq 1)$ be the sequence of `rewards` 
when process is started from a `state-action pair` $(S_{1}, \ A_{1})$
drawn from a positive probability distribution over $\mathcal{X} \times \mathcal{A}$ 
and follows the policy $\pi$ for $t\geq 2$.

Then,
$$
Q^{\pi}(s,a)
\triangleq \mathbb{E}\left[ \sum ^{\infty}_{t=1}
\gamma^{t-1} R_{t} \mid S_{1} = s, \ A_{1}=a \right]
$$

> The `action-value function` $Q^{\pi}$ evaluated at $(s,a)$ is the expected discounted return when the agent starts at state $s$, takes action $a$ and then follows policy $\pi$.

---
## Relation to Value Function
Compared to the [[Value Function|value function]], the difference is that the first action $A_{1}$ in $V^{\pi}$ is selected according to policy $\pi(\cdot \mid S_{1})$, but the first action in $Q^{\pi}(s,a)$ is pre-defined action $a$.

So,
$$
\begin{align}
V^{\pi}(s) 
&= \mathbb{E}[Q^{\pi}(s,A)] \\[6pt]
&= \int \pi(da \mid s) \ Q^{\pi}(s,a)
\end{align}
$$


