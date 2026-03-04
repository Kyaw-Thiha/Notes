# Naive Policy Evaluation
#rl/planning/naive

Given an [[Markov Decision Process (MDP)|MDP]] $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ and a [[Markov Policy|policy]] $\pi$, we would like to compute $V^{\pi}$ or $Q^{\pi}$.

$$
V^{\pi}(s)
= \mathbb{E} \left[ \sum^\infty_{t=1} \ \gamma^{t-1}
R_{t} \mid S_{t} = s \right]
$$

We can represent all possible states & actions in a tree structure.

![image|600](https://notes-media.kthiha.com/Policy-Evaluation/d07b010fb8ff6342bc1e5d99b9ae52ae.png)

For example at time $t=2$, the probability of an agent being in state $s'$ and choosing action $a'$ is
$$
\sum_{a \in \mathcal{A}} \pi(a \mid s)
\ \mathcal{P}(s' \mid s,a) \ \pi(a' \mid s')
$$
Hence, the expected reward at $t=2$ is
$$
\sum_{a, \ s', \ a'} \pi(a \mid s)
\ \mathcal{P}(s' \mid s,a) \ \pi(a' \mid s')
\ r(s' \mid a')
$$

We can then add them in a discounted way to get $V(s)$.

`Drawbacks`
- Size of tree grows exponentially fast.
- For [[Markov Decision Process (MDP)|continuing MDP]], the depth of the tree is infinity.

---
## See Also
- [[Markov Decision Process (MDP)]]
- [[Markov Policy]]
