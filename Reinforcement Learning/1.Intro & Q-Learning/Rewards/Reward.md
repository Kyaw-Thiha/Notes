# Reward
#rl/reward

The `expected reward` is the foundational objective an agent seeks to maximize.

![Expected Reward|400](https://huggingface.co/blog/assets/63_deep_rl_intro/rewards_4.jpg)

---
## Defining the expected reward
To guide the agent maximizing the performance, we can define `expected reward` as
$$
r(s, a)
\triangleq \mathbb{E} \ [R \mid S=s, A=a]
$$
where the random variable $R$ is distributed according to $\mathcal{R}(\cdot \mid s, a)$.

> Note that the `expected reward` is averaged across multiple [[Episode|episodes]].

---
## Maximizing Expected Reward
At state $s$, the agent should chooses an action $a$ that maximizes the average reward $r(s, a)$ at that state.
$$
a^{*}
\leftarrow \arg \max_{a \in \mathcal{A}} \ r(s, a)
$$
By definition of $\arg\max$, no choice of action can gather more rewards in expectation.

Hence, we can define the optimal policy $\pi^{*}: \mathcal{X} \to \mathcal{A}$ as 
$$
\pi^{*}(x)
\leftarrow \arg \max_{a \in \mathcal{A}}
\ r(s, a)
$$

---
## Reward across Finite Horizon Tasks
**Finite Horizon Task**: The agent interacts with the environment for a fixed $T \geq 1$ number of steps.

For each [[Episode|episode]],
- The agent starts at $S_{1} \sim \rho \in \mathcal{M}(\mathcal{X})$
- It chooses an action $A_{1} \sim \pi(\cdot \mid S_{1})$
- The agent goes to the next state $S_{2} \sim \mathcal{P}(\cdot \mid S_{1}, A_{1})$.
  And receives the reward $R_{1} \sim \mathcal{R}(\cdot \mid S_{1}, A_{1})$.
- This process repeats until $\dots$
- The agent gets to the terminal state $S_{T} \sim \mathcal{P}(\cdot \mid X_{S-1}, \ A_{T-1})$.
  It chooses action $A_{T} \sim \pi(\cdot \mid S_{T-1})$.
  It then receives reward $R_{T} \sim \mathcal{R}(\cdot \mid S_{T}, A_{T})$.

---
## Return of a Policy
To measure the performance, we can compute the sum of rewards:
$$
G^{\pi}
\triangleq R_{1} + \dots + R_{T}
$$
The random variable $G^{\pi}$ is called the `return` of [[Policy|policy]] $\pi$.

## Discounted Sum of Rewards
In order to prioritize earlier rewards, we can define a discounted sum of rewards as the `return`
$$
G^{\pi}
\triangleq R_{1} + \gamma R_{2} + \dots + \gamma^{T - 1} R_{T}
$$
where 
- $0 \leq \gamma \leq 1$ is the `discount factor`


---

