# SARSA
[[SARSA]] can be thought of as a [[Stochastic Approximation(SA)|stochastic approximation]] of the [[Quality Function|Q-function]] by following a [[Policy|policy]] $\pi_{t}$ that is close to [[Greedy Policy|greedy policy]] $\pi_{g}$ but with exploration, and on-policy.

$$
Q_{t+1}(S_{t}, A_{t})
\leftarrow (1 - \alpha_{t}(S_{t}, A_{t}))
\ \underbrace{Q_{t}(S_{t}, A_{t})}_{\text{Current Estimate}}
+ \alpha_{t}(S_{t}, A_{t}) \underbrace{
\left[ R_{t} + \gamma 
Q_{t}(S_{t+1}, A_{t+1}) \right]}_{\text{Return based on policy } \pi_{t}}
$$


---
## Main Idea
Compared to [[Q-Learning]], we can also have a [[Policy Iteration|Policy Iteration-like]] procedure: 
> Estimate [[Quality Function|Q-function]] $Q^{\pi}$ for a given [[Policy|policy]] $\pi$ 
> Perform [[Policy Iteration|policy improvement]] to obtain a new $\pi$

- **Usual PI**
  Wait long enough until the [[Temporal Difference Learning for Policy Evaluation(TD)|TD algorithm]] produces a $Q \to Q^{\pi}$, then improve
- **Generalized Policy Iteration**
  Improve the [[Policy|policy]] $\pi$ before $Q$ converges to $Q^{\pi}$.

---
## SARSA Algorithm
- At state $S_{t}$, the agent chooses $A_{t} = \pi_{t}(S_{t})$.
- It then receives $S_{t+1} \sim \mathcal{P}(\cdot \mid S_{t}, A_{t})$ and $R_{t}\sim \mathcal{R}(\cdot \mid S_{t}, A_{t})$.
- At timestep $t+1$, it chooses $A_{t+1}=\pi_{t}(S_{t+1})$ and updates the [[Quality Function|action-value function]] estimate as 

$$
\boxed{ \ Q_{t+1}(S_{t}, A_{t})
\leftarrow (1 - \alpha_{t}(S_{t}, A_{t}))
\ Q_{t}(S_{t}, A_{t})
+ \alpha_{t}(S_{t}, A_{t})
\left[ R_{t} + \gamma 
Q_{t}(S_{t+1}, A_{t+1}) \right] \ }
$$
for the observed $(S_{t}, A_{t})$ and $Q_{t+1}(s,a) \leftarrow Q_{t}(s,a)$ for all other states $(s,a) \neq (S_{t}, A_{t})$.

> Note that the [[Policy|policy]] $\pi_{t}$ is often chosen as close to [[Greedy Policy|greedy policy]], but with some amount of exploration: [[Epsilon Greedy Policy]].
> 
> The [[Greedy Policy|greedy part]] performs the [[Policy Iteration|policy improvement]], while the
occasional random choice of actions allows the agent to have some exploration.

---
## Comparism to Q-Learning
Comparing the update rules:
- [[Q-Learning]]: $\max_{a' \in \mathcal{A}} Q_{t}(S_{t+1}, a')$
- [[SARSA]]: $Q_{t}(S_{t+1}, A_{t+1}) = Q_{t}(S_{t+1}, \pi_{t}(S_{t+1}))$

Comparing the evaluated policy:
- [[Q-Learning]]: the [[Greedy Policy|greedy policy]] $\pi_{g}(Q_{t})$ (off-policy)
- [[SARSA]]: the same [[Policy|policy]] $\pi_{t}$ that selects actions (on-policy)

---
## See Also
- [[Q-Learning]]
- [[Policy Iteration]]
- [[Greedy Policy]]
- [[Epsilon Greedy Policy]]
- [[Policy]]