## Introduction
The [[Online Learning of the Reward Function|reward learning problem]] is a special case of [[Value Function|value function]] learning where the [[Episode|episode]] ends in one time-step.

Now, we will devise methods to learn or estimate the [[Value Function|value function]] $V^{\pi}$ and $Q^{\pi}$ of a policy, and then the optimal value function $V^{*}$ and $Q^{*}$.

---
## Problem Formulation
Recall from [[Value Function|here]] that 
$$
V^{\pi}(s)
= \mathbb{E}[G_{t}^{\pi} \mid S_{t} =  s]
$$
with
$$
G^{\pi}_{t} \triangleq \sum_{k\geq t}
\gamma^{k-t} \ R_{k}
$$

So, $G^{\pi}_{t}$ conditioned on starting from $S_{t}=s$ plays the same rule as the $r.v.$ $Z$ in [[Online Estimator of Mean of Random Variable|estimating]] $m=\mathbb{E}[Z]$.

In order to obtain sample from [[Reward|return]] $G^{\pi}$,
- Suppose the agent starts at state $s$ an follows $\pi$
- We can draw one sample of $r.v.$ $G^{\pi}$
- This is done by computing the discounted sum of rewards collected during the episode.

Each trajectory is called a `rollout`.
Estimation methods based on the whole trajectory or rollouts is called Monte Carlo estimates.

---
If we repeat this process from the same state, we get another draw of $r.v.$ $G^{\pi}$.
Let's call the [[Value Function|value]] of these samples as
$$
G^{\pi(1)}(s), \ G^{\pi(2)}(s), \ \dots, 
\ G^{\pi(n)}(s)
$$
We can get an estimate $\hat{V}(s)$ of $V^{\pi}(s)$ by taking the [[Sample Average Estimator|sample average]]:
$$
\hat{V}^{\pi}(s)
= \frac{1}{n} \sum^{n}_{i=1} G^{\pi(i)}(s)
$$
We can also use the [[Stochastic Approximation(SA)|stochastic approximation]] too.