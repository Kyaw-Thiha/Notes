# Finite Policy Parameter Space
#rl/policy-methods/zero-order
Assume we are given finite $\Theta=\{ \theta_{1}, \ \dots, \ \theta_{m} \}$ [[Policy Parameterization|policy parameters]].
This defines the [[Policy|finite policy space]] $\Pi_{\Theta} = \{ \pi_{\theta}: \theta \in \Theta \}$.

> **Goal**:
> Find a policy $\pi_{\theta} \in \Pi_{\Theta}$ such that [[Policy Search Performance Measure|performance]] $J_{\rho}(\pi_{\theta})$ is maximized.

If we can easily compute $J_{\rho}(\pi_{\theta})$ for each $\theta \in \Theta$, we can find such maximization.
So, the main issue is how to compute $J_{\rho}(\pi_{\theta})$.

---
## Direct Policy Evaluation
Recall that the performance measure is
$$
J_{\rho}(\pi_{\theta})
= \mathbb{E}_{X \sim \rho} 
[ \ V^{\pi_{\theta}}(S) \ ]
$$
which is the expectation of $V^{\pi_{\theta}}(S)$ $w.r.t$ $S \sim \rho$.

Using any of the prior [[Planning (Reinforcement Learning)|policy evaluation]] methods,
- we can try to compute $V^{\pi_{\theta}}(s)$ for all $s \in \mathcal{S}$
- and take the weighted average according to $\rho$.

If state space $\mathcal{S}$ is discrete, this would be
$$
J_{\rho}(\pi_{\theta})
= \sum_{s \in \mathcal{S}} \rho(x) 
\ V^{\pi_{\theta}}(s)
$$

Note that if state space $\mathcal{S}$ is large,
- computing $V^{\pi_{\theta}}$ itself is not going to be easy
- computing integral $\int V^{\pi_{\theta}}(s) \ d\rho(s)$ is going to challening

---
## Unbiased Estimate
An alternative would be computing an unbiased estimate of $J_{\rho}(\pi_{\theta})$ instead using [[Monte Carlo Methods|Monte Carlo estimation]].
- Assume that we know $V^{\pi_{\theta}}$. Estimate $J_{\rho}(\pi_{\theta})$.
- Replace $V^{\pi_{\theta}}(s)$ with the return $G^{\pi_{\theta}}(s)$.

---
### First Step
Assume that we know $V^{\pi_{\theta}}$ and we want to estimate $J_{\rho}(\pi_{\theta})$.
If we sample $S \sim \rho$, we have 
$$
\mathbb{E}[V^{\pi_{\theta}}(S)]
= \int V^{\pi_{\theta}}(s) \ d\rho(s)
= J_{\rho}(\pi_{\theta})
$$
where $V^{\pi_{\theta}}(S)$ is an unbiased estimate of $J_{\rho}(\pi_{\theta})$.c

If we draw $n$ independent samples $S_{1}, \ \dots, \ S_{n} \sim \rho$, the estimator
$$
\frac{1}{n} \sum ^{n}_{i=1} V^{\pi_{\theta}}(S_{i})
$$
would be unbiased as well with variance of
$$
\frac{Var[V^{\pi_{\theta}}(S)]}{n}
$$
which goes to $0$ as $n$ increases.

**Variance**:
The variance $Var[V^{\pi_{\theta}}(S)]$ is a measure of dispersion of the
[[Value Function|value function]] across states samples according to $\rho$.
- If the value function is constant, it is $0$.
- If the value function is changing slowly, it would be small.
- If the value function varies greatly, the variance is large.

The variance is a function of the [[Policy|policy]] $\pi_{\theta}$ so for each $\pi_{\theta} \in \Pi_{\Theta}$, we get a different variance.

---
### Second Step
Replace [[Value Function|value function]] $V^{\pi_{\theta}}(s)$ with the [[Reward|return]] $G^{\pi_{\theta}}(s)$.
where $G^{\pi_{\theta}}$ is the unbiased estimate of $V^{\pi_{\theta}}$.

Computation of $G^{\pi_{\theta}}(s)$ requires 
- starting the agent from state $s$ and 
- following $\pi_{\theta}$ until the end of episode for episodic tasks or till infinity for continual tasks.

If $S \sim \rho$, $G^{\pi_{\theta}}(S)$ is an unbiased estimate of $J_{\rho}(\pi_{\theta})$ as
$$
\begin{align}
\mathbb{E}_{S \sim \rho}[G^{\pi_{\theta}}(S)]
&= \mathbb{E}_{S \sim \rho}[ \ \mathbb{E} 
[G^{\pi_{\theta}}(S) \mid S] \ ] \\[6pt]
&= \mathbb{E}_{S \sim \rho}[V^{\pi_{\theta}}(S)] 
\\[6pt]
&= J_{\rho}(\pi_{\theta})
\end{align}
$$

If we draw $n$ independently selected $S_{1}, \ \dots, \ S_{n} \sim \rho$, we can form
$$
\hat{J}_{n}(\pi_{\theta})
= \frac{1}{n} \sum ^{n}_{i=1} 
G^{\pi_{\theta}}(S_{i})
$$
which is an unbiased estimate of $J_{\rho}(\pi_{\theta})$.

---
> **Proposition**
> The estimator $\hat{J}_{n}(\pi_{\theta})$ is an unbiased estimate for $J_{\rho}(\pi_{\theta})$ and has the variance of

$$
Var[\hat{J}_{n}(\pi_{\theta})]
= \frac{1}{n} ( \ \mathbb{E}[ 
Var[ \ G^{\pi_{\theta}}(S) \mid S \ ]]
+ Var[V^{\pi_{\theta}}(S)] \ )
$$

---
If we have a finite number of parameters in $\Theta$, we can estimate
$$
J_{\rho}(\pi_{\theta_{i}})
\approx \hat{J}_{n}(\pi_{\theta_{i}})
\pm O_{p}\left( \frac{1}{\sqrt{ n }} \right)
$$
for each $\theta_{i} \in \Theta$.

Here $O_{P}(\cdot)$ is an order notation that hides quantities related to “this statement holds with probability at least $1 - \delta$".

We can then use these estimates to select the best among them:
$$
\hat{\pi} 
= \pi_{\hat{\theta}} \leftarrow \arg\max_{\theta 
\in \Theta} \hat{J}_{n}(\pi_{\theta})
$$
This can be done with $n|\Theta|$ rollouts.

---
#### Estimation Error
As there is an $O_{P}\left( \frac{1}{\sqrt{ n }} \right)$ error in estimation of each $J_{\rho}(\pi_{\theta})$,
the selected [[Policy|policy]] $\hat{\pi}$ may not be same as as the maximizer $\bar{\pi}$.

A mistake in the choice of [[Policy|optimal policy]] happens if
$$
\hat{J}_{n}(\hat{\pi})
> \hat{J}_{n}(\bar{\pi})
$$
which leads to preferring $\hat{\pi}$ to $\bar{\pi}$ according to [[Policy Search Performance Measure|empirical performance measure]] even though
$$
J_{\rho}(\hat{\pi})
< J_{\rho}(\bar{\pi})
$$

- Even if we make an error in selecting the [[Policy|best policy]], the gap in their performance within $O_{P}\left( \frac{1}{\sqrt{ n }} \right)$.
- As we increase $n$, the error in estimating $J_{\rho}(\pi_{\theta})$ decreases. 
  And the probability of selecting an [[Policy|optimal policy]] increases.
- This increased accuracy however is at the cost of increased sample and computational complexity.
  Leading to $n|\Theta|$ rollouts.

---
> **Proposition**
> Consider $\hat{\pi} = \pi_{\hat{\theta}} \leftarrow \arg\max_{\theta \in \Theta} \hat{J}_{n}(\pi_{\theta})$.
> Assume that the [[Reward|returns]] $G^{\pi_{\theta}}(s)$ are all [[Quality Function|quality function]] $Q_{max}\text{-bounded}$ for any $\theta \in \Theta$ and $s \in \mathcal{S}$.
> Furthermore, suppose that $|\Theta| < \infty$.
> For any $\delta > 0$, we have that

$$
J_{\rho}(\hat{\theta})
\geq \max_{\theta \in \Theta} J_{\rho}(\theta)
- 2Q_{max} \sqrt{ \frac{2 \ln\left( \frac{2|\Theta|}
{\delta} \right)}{n} }
$$
> with probability of at least $1 - \delta$.

---
## Infinite Parameter Space
If $\Theta$ is not finite, we cannot evaluation $\hat{J}_{n}(\pi_{\theta})$ for all $\theta \in \Theta$.
Instead, we could use different techniques like
- [[Random Search (Zero-Order)|Random Search]]
- Simulated Annealing
- [[Evolutionary Algorithms RL|Evolutionary Algorithms]]

---
## See Also
- [[Finite Policy Parameter Space (Zero-Order)]]
- [[Random Search (Zero-Order)]]
- [[Evolutionary Algorithms RL]]
- [[Policy Search Performance Measure]]
- [[Policy Parameterization]]