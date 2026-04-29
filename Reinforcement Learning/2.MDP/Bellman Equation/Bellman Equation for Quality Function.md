# Bellman Equation
#rl/bellman-equation/quality-function

The `Bellman Equation` for the [[Quality Function|Q-function]] $Q^{\pi}$ represents the `action-value function` in reinforcement learning as  immediate reward and the discounted future value.

![Bellman Equation](https://miro.medium.com/v2/resize:fit:1400/1*5sWWgc2ReqdL3nmjrEF2PA.png)

---
## Bellman Equation for Q-Function
For a policy $\pi$, the `Bellman equation` for the action-value function $Q^{\pi}$ is
$$
\begin{align}
&Q^{\pi}(s,a) \\[6pt]
&= r(s,a) + \gamma \int \mathcal{P}(ds' \mid s, a)
\ V^{\pi}(s') \\[6pt]
&= r(s, a) + \gamma \int \mathcal{P}(ds' \mid s, a)
\ \pi(da' \mid s') \ Q^{\pi}(s', a')
\end{align}
$$

Using [[Value Function|value function]] $V^{\pi}(x) = \int \pi(da \mid s) \ Q^{\pi}(s,a)$, we can get
$$
Q^{\pi} = r + \gamma \ \mathcal{P} V^{\pi}
$$

> **Remark**: Compared to the [[Bellman Equation for Value Function]], the choice of action at the first time step is pre-specified instead of being selected by policy $\pi$.

---
## See Also
- [[Quality Function]]
- [[Value Function]]
- [[Bellman Equation for Value Function]]
- [[Bellman Equation for Optimal Quality Functions]]
- [[Greedy Policy]]