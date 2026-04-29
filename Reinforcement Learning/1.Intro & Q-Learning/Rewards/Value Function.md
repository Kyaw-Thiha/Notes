# Value Function
#rl/value-function
A `value function` estimates the expected cumulative future [[Reward|reward]] an agent can achieve from a specific state $s$ or state-action pair $(s, a)$, while following a [[Policy|policy]] $\pi$.

![Bellman Equation|500](https://machinelearningmastery.com/wp-content/uploads/2024/07/rl-bellman-equation-mlm.png)

---
## Defining the Value Function
Recall that the [[Reward|discounted sum of rewards]] is a random variable.

To define a performance measure that is not random, the `value function` $V^{\pi}: \mathcal{X} \to \mathbb{R}$ compute its expectation:
$$
V^{\pi}(s)
\triangleq \mathbb{E} \left[ \sum^T_{t=1}
\gamma^{t-1} R_{t} \mid S_{1} = s \right]
$$

This is the expected value of [[Reward|return]] if the agent starts at state $s$ and follows policy $\pi$.

---
## Value function over an episode
We can define the return from time $\tau \in \{ 1, \ \dots, \ T \}$ until the end of the [[Episode|episode]] $T$ as
$$
G^{\pi}_{\tau}
\triangleq \sum^T_{t=\tau} \ \gamma^{t - \tau} R_{t}
$$

Using it, we can define the `value function` as
$$
V^{\pi}_{\tau}
\triangleq \mathbb{E}[ \ G^{\pi}_{\tau} 
\mid S_{\tau} = s  \ ]
$$

---
## Relation to Reward
At $T=1$, we get that
$$
V^{\pi}(x)
= \mathbb{E}[R_{1} \mid S_{1} = s]
$$
Note that this is similar to the [[Reward|reward function]]
$$
r^{\pi}(s) 
= \mathbb{E}[R \mid S=s]
$$
parameterised with the action determined by policy $A \sim \pi(\cdot \mid s)$.

---
## Finding Optimal Policy
Denote the space of all stochastic policies as
$$
\Pi
= \{ \pi: \pi(\cdot \mid s) \in 
\mathcal{M}(\mathcal{A}), 
\ \forall s \in \mathcal{S} \}
$$
Then, we need to find
$$
\pi^{*} \leftarrow \arg\max_{\pi \in \Pi} V^{\pi}
$$
To find such optimal policy $\pi^{*}$, we need to search over all possible policies.

---
### Formalizing Optimal Policy

**Defining Optimal Policy**
Let $\pi$ and $\pi'$ be the two [[Policy|policies]] being compared against.
Assume they are `Markov stationary` such that $A_{t} \sim \pi_{t}(\cdot \mid S_{t})$.

> Then, we can say $\pi \geq \pi'$ $\text{iff}$ $V^{\pi}(s) \geq V^{\pi'}(s)$, $\forall s \in \mathcal{S}$.

![Optimal Policy|300](https://notes-media.kthiha.com/Value-Function/2e3b8ed98032241b0e4c0742a52187b6.png)

> We can also use strict inequality $\pi > \pi'$ if $V^{\pi}(s) \geq V^{\pi'}(s), \ \forall s \in \mathcal{S}$  and $\exists s' \in \mathcal{S}$ s.t. $V^{\pi}(s') > V^{\pi'}(s')$.

**Optimal Policy**
If we can find a policy $\pi^{*}$ that satisfies $\pi^{*} \geq \pi, \ \forall \pi$, we call it an `optimal policy`.

---
### Decomposing Value Function

For each states $s \in \mathcal{S}$,
$$
\begin{align}
V^{\pi}(s)
&= \int \mathcal{R}(dr \mid s, a) \ \pi(da \mid s)  
\\[6pt]
&= \int_{A} \left( \ \int_{R} \ r  
\mathcal{R}(dr \mid s,a)  
\ \right) \ \pi(da \mid s) \\[6pt]
&= \int \pi(da \mid s) \int \mathcal{R}(dr \mid s, a)
&\text{by (1)}
\\[6pt]
&= \int \pi(da \mid s) \ r(s, a)
&\text{by (2)}
\end{align}
$$
where
- $\mathcal{R}(\cdot \mid s,a)$ is the `probability distribution over rewards`
- $\pi(\cdot \mid s)$ is the `probability distribution over actions`
- $V^{\pi}(s)$ is the `expected reward` when 
  1. you sample $a \sim \pi(\cdot \mid s)$
  2. then sample $r \sim \mathcal{R}(\cdot \mid s, a)$

and 
- $(1)$: by `Fubini's Theorem`
- $(2)$: by $r(s,a) \overset{\text{def}}{=} \int_{R} r \ \mathcal{R}(dr \mid s, a)$

---
### Finding optimal policy at T=1
At $T=1$, the values of $V^{\pi}$ at two different states $s_{1}, s_{2} \in \mathcal{S}$ do not have any interaction with each other.
Hence, we can find the optimal policy at each state separately.

Based on above decomposition, finding an optimal $\pi(\cdot \mid s)$ that maximises $V^{\pi}(s)$ means
$$
\sup_{\pi(\cdot \mid s) \in \mathcal{M}
(\mathcal{A})}
\int \pi(da \mid s) \ r(s, a)
$$
The maximum distribution concentrates all its mass at the optimal action $a^{*}$ that maximizes $r(s,a)$, assuming it exists.
Hence,
$$
\pi^{*}(a \mid s)
= \delta(a - \arg\max_{a' \in \mathcal{A}} r(s,a'))
$$
or equivalently,
$$
\pi^{*}(s)
= \arg\max_{a \in \mathcal{A}} r(s,a)
$$

> In other words, since the chosen action $a$ define the best reward $r(\cdot \mid s,a)$ given a state $s$, 
> choosing the optimal action $a^{*}$ is equivalent to choosing the best value function $V^{\pi}(s)$ at $T=1$.

---
**Sidenote**

Throughout this page, we are assuming that [[Policy]] is `stationary`.
But note that it can also be `non-stationary` $\bar{\pi} = (\pi_{1}, \ \dots, \ \pi_{T})$.

---
## Norm of Value Function
For $V \in \mathcal{B}(\mathcal{S})$, its [[Infinity Norm|supremum norm]] is
$$
||V||_{\infty}
= \sup_{s \in \mathcal{S}} |V(s)|
$$
This is the maximum value of the value function $V$ over the state space.

---
## See Also
- [[Reward]]
- [[Quality Function]]
- [[Policy]]
- [[Bellman Equation for Value Function]]
