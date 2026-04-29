# Policy Gradient

---
## Discounted Future-State Distribution
Recall that 
$$
\mathcal{P}^{\pi}(\cdot \mid s; k)
= \mathcal{P}^{\pi}(\cdot \mid s)^{k}
$$
is the probability distribution of following $\pi$ for $k \geq 0$ steps.

Let discounted future-state distribution of starting $s \in \mathcal{S}$ and following $\pi$ be
$$
\rho^{\pi}_{\gamma}(\cdot \mid s)
= \rho_{\gamma}(\cdot \mid s; \ \mathcal{P}^{\pi})
\triangleq (1 - \gamma) \sum_{k \geq 0}
\gamma^{k} \mathcal{P}^{\pi}(\cdot \mid s; k)
$$

The relevance of this distribution becomes more clear when we note that
$$
\begin{align}
V^{\pi}(s)
&= \mathbb{E}\left[ \sum_{t\geq 0} \gamma^{t} R_{t} 
\mid S_{0} = s \right] \\[6pt]
&= \sum_{t \geq 0} \gamma^{t} \mathbb{E}[R_{t} \mid  
S_{t} = s] \\[6pt]
&= \sum_{t \geq 0} \gamma^{t}
\int \mathcal{P}^{\pi}(ds' \mid s; t) \ r(s') \\[6pt]
&= \frac{1}{1 - \gamma} \int \rho^{\pi}_{\gamma} 
(ds' \mid s) \ r(s') \\[6pt]
&= \frac{1}{1 - \gamma} \mathbb{E}_{S'  
\sim \rho^{\pi}_{\gamma}(\cdot \mid s)}
[r(S')]
\end{align}
$$
> **Interpretation**:
> The agent starts from state $s$ and at each time step, it decides to follow $\pi$ with the probability of $\gamma$ or terminates the episode with probability $1- \gamma$.

We can also define discounted future-state distribution of starting from $\rho$ and following $\pi$ as
$$
\rho^{\pi}_{\gamma}(\cdot)
= \rho_{\gamma}(\cdot \mid \mathcal{P}^{\pi})
\triangleq \int \rho_{\gamma}(\cdot \mid s; \ \mathcal{P}^{\pi}) \ d\rho(s)
$$

The [[Policy Search Performance Measure|performance measure]] $J(\pi_{\theta})$ is
$$
J(\pi_{\theta}) 
= \mathbb{E}_{S \sim \rho} [V^{\pi_{\theta}}(S)]
= \frac{1}{1 - \gamma} \mathbb{E}_{S \sim \rho^{\pi}_{\gamma}} [r(S)]
$$
---
## Policy Gradient Theorem
Using [[Policy Gradient Theorem]], we can relate [[Policy Gradient for Continuing Tasks|policy gradient]] to
- the discounted future-state distribution $\rho^{\pi_{\theta}}_{\gamma}$
- the [[Quality Function|action-value function]] $Q^{\pi_{\theta}}(s,a)$
- and the gradient of $\pi_{\theta}$

We can now improve the [[policy]] by performing [[Gradient Descent|gradient ascent]] 
$$
\begin{align}
&\theta_{k+1} \leftarrow \theta_{k}  
+ \alpha_{k} \nabla_{\theta} J_{\rho} 
(\pi_{\theta_{k}}) \\[6pt]
\iff &\theta_{k+1} \leftarrow \theta_{k}  
+ \mathbb{E}_{S \sim \rho^{\pi_{\theta}}_{\gamma},  
A \sim \pi_{\theta}(\cdot \mid S)} [\nabla_{\theta} 
\log \pi_{\theta}(A \mid S) \ Q^{\pi_{\theta}}(S,A)]
\end{align}
$$

---
## Sampling $S$ from $\rho^{\pi_{\theta}}_{\gamma}$
Sampling from $\rho^{\pi_{\theta}}_{\gamma}$ is relatively straightforward in the on-policy sampling scenario when the agent follows $\pi_{\theta}$.
- the agent starts an episode from $S_{0} \sim \rho$ and follows $\pi_{\theta}$.
- we get a sequence of states $S_{0}, \ S_{1},  \dots$
- these would be samples from $\int d\rho(s) \mathcal{P}^{\pi_{\theta}}(\cdot \mid s;k)$ for $k=0,1,\dots$
- The distribution $\rho^{\pi_{\theta}}_{\gamma}$ however has a $\gamma^{k}$ factor for the $k^{th}$ timestep
- Its effect is that the contribution to the gradient from $S_{k}$, which is the following, should be weighted by $\gamma^{k}$:

$$
\mathbb{E}[\nabla_{\theta} \log \pi_{\theta}
(A \mid S) \ Q^{\pi_{\theta}}(S,A)]
= \int \pi_{\theta}(a \mid s) \
\nabla_{\theta}\pi_{\theta}(a \mid s)
Q^{\pi_{\theta}}(s,a) da
$$

Alternatively, we can directly sample from $\rho^{\pi_{\theta}}_{\gamma}$ by following $\pi$ but at each time step, terminates the episode with probability $1 - \gamma$.

---
## Sampling $A$ from $\pi_{\theta}(\cdot \mid S)$
An action $A$ sampled from $\pi_{\theta}(\cdot \mid S)$ is automatically generated when the agent follows [[policy]] $\pi_{\theta}$.

---
## Computation of $Q^{\pi_{\theta}}(S, A)$
The remaining issue is the computation of $Q^{\pi_{\theta}}(S,A)$ for $S \sim \rho^{\pi_{\theta}}_{\gamma}$ and $A \sim \pi_{\theta}(\cdot \mid S)$ using data.

This is essentially a [[Planning (Reinforcement Learning)|policy evaluation]] problem and we may use various [[Quality Function|action-value function]] estimators.

### REINFORCE
Simple approach would be using [[Monte Carlo Estimation for Policy Evaluation|monte carlo estimate]] of $Q^{\pi_{\theta}}(S,A)$.

- In the on-policy setting when the agent follows $\pi_{\theta}$, it generates the sequence $S_{0}, A_{0}, R_{0}, \ S_{1}, A_{1}, R_{1}, \ \dots$ with $A_{t} \sim \pi_{\theta}(\cdot \mid S_{t})$.
- The [[Reward|return]] $G_{t}^{\pi} = \sum_{k \geq t} \gamma^{k-t} R_{k}$ is an unbiased estimate of $Q^{\pi_{\theta}}(S_{t}, A_{t})$.
- We replace the [[Quality Function|action-value function]] at that state-action with its return from time $t$ onwards.
- The [[Reward|return]] however is a high-variance estimate of the [[Quality Function|action-value function]].
- We can use a baseline to reduce the variance of this [[Monte Carlo Estimation for Control|Monte Carlo estimate]].

---
### Actor-Critic Methods
Another approach is to use an [[Quality Function|action-value function]] estimator.
One such method is called `actor-critic method`.
- The actor refers to the [[policy]]. 
  And often [[Policy Gradient for Continuing Tasks|policy gradient]] is used to improve it.
- The critic refers to the [[value function]] estimator used to criticize the [[policy]].

The use of critic may induce a bias as $\mathbb{E}[\hat{Q}^{\pi_{\theta}}(S,A) \mid S,A]$ may be different from $Q^{\pi_{\theta}}(S,A)$.
This is especially true if 
- we use a [[Temporal Difference Learning for Policy Evaluation(TD)|TD method]]
  (which introduces bias due to bootstrapping)
- or a [[Value Function Approximation|function approximator]] 
  (for large state-action spaces)

---
