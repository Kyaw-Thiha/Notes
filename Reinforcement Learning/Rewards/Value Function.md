# Value Function
#rl/value-function
A `value function` estimates the expected cumulative future [[Reward|reward]] an agent can achieve from a specific state $s$ or state-action pair $(s, a)$, while following a [[Markov Policy|policy]] $\pi$.

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
