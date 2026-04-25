# Markov Decision Process
#rl/markov-decision-process
`Markov Decision Process (MDP)` is a mathematical framework use to model sequential decision-making.

![MDP|400](https://notes-media.kthiha.com/Markov-Decision-Process-(MDP)/1bac4039479d810db0c24420a5c5014b.png)

Note that outcomes are partly random and partly controlled by the agent.

---
**Formal Definition**
A `discounted MDP` is a $5\text{-tuple}$ $(\mathcal{X}, \ \mathcal{A}, \ \mathcal{P}, \ \mathcal{R}, \ \gamma)$ where
- $\mathcal{X}$ is the `state space`
- $\mathcal{A}$ is the `action space`
- $\mathcal{P}: \mathcal{X} \times \mathcal{A} \to \mathcal{M}(\mathcal{X})$ is the `transition probability kernal`
- $\mathcal{R}: \mathcal{X} \times \mathcal{A} \to \mathcal{M}(\mathbb{R})$ is the `immediate reward distribution`
- $0 \leq \gamma < 1$ is the `discount factor`

---
## Markov Decision Process
- The `initial state` is drawn from the initial state distribution $\rho \in \mathcal{M}(\mathcal{X})$:

$$
X_{1} \sim \rho \ , \quad \text{where } \rho \in \mathcal{M}(\mathcal{X})
$$
- The agent chooses an `action` from the action space $A_{t} \in \mathcal{A}$.
  It is usually sampled from the [[Policy|policy]] $\pi$:

$$
A_{t} \sim \pi(\cdot \mid X_{t})
$$

- The agent's `state` is updated to $X_{t+1}$.

$$
X_{t+1} \sim \mathcal{P}(\cdot \mid X_{t}, \ A_{t})
$$
- The agent then receives the `reward` $R_{t}$.
$$
R_{t} \sim \mathcal{R}(\cdot \mid X_{t}, A_{t})
$$
- This process repeats.

**Trajectory**
The `trajectory` can be denoted as
$$
\xi = (X_{1}, A_{1}, R_{1}, \ X_{2}, A_{2}, R_{2}, \ \dots)
$$
The `space of all trajectories` can be denoted as $\Xi$.

---
### State Space & Dynamics
**State Space**
The `state space` can be
- Finite: $\mathcal{X} \ \{ x_{1}, \ x_{2}, \ \dots, \ x_{n} \}$ with $n < \infty$
- Infinite but countable: $\mathcal{X} = \{ x_{1}, \ x_{2}, \ \dots \}$
- Continuous: $\mathcal{X} \subset \mathbb{R}^d$

**Dynamics**
The `dynamics` can be
- Stochastic
- Deterministic

---
## See Also
- [Main Reference Slide](https://amfarahmand.github.io/IntroRL/lectures/lec01.pdf)
- [Reference Book](https://amfarahmand.github.io/IntroRL/lectures/FRL.pdf)
- [[Policy]]
- [[Policy-Induced Transition Kernel]]