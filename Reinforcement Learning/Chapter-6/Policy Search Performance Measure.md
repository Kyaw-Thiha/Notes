# Performance Measure
#rl/policy-methods/performance-measure
> Focus on [[Reward|expected return]] of following [[Policy|policy]] $\pi_{\theta}$, which is the [[Value Function|value function]].
> Incorporate the variance or some other risk measures too.

**Goal**: Find the [[Policy|policy]] that maximizes this performance measure.
**Constraint**: We are restricted to choosing [[Policy|policies]] within $\Pi_{\Theta}$.

---
## Single State
Assume that we only care about the performance at state $s \in \mathcal{S}$.
Then, the goal of the policy search is
$$
\arg\max_{\pi \in \Pi_{\theta}} V^{\pi}(s)
= \arg\max_{\theta \in \Theta} V^{\pi_{\theta}}(s)
$$
> Find a policy such that if the agent starts at particular state $s$, its performance measured according to [[Reward|expected return]] is maximized.

- The [[Policy|optimal policy]] $\pi^{*}$ not only maximizes the [[Value Function|value function]] at this particular $s$ but also at any other $s' \in \mathcal{S}$.
- But note that the [[Policy|optimal policy]] $\pi^{*}$ may not be in $\Pi_{\Theta}$.
- If $\pi^{*} \notin \Pi_{\Theta}$, we will not be able to find a [[Policy|policy]] that maximizes the value at all states.
	- In that case, find a [[Policy|policy]] that is only good at our starting state $s$.
	  Ignore the performance at other states.
	- The obtained [[Policy|policy]] is going to be initial-state-dependant.
	  If we change $s$ to another state $s' \neq s$, the [[Policy|optimal policy]] within $\Pi_{\Theta}$ might change.

---
## Initial State Distribution $X_{1} \sim \rho$
Instead of considering a single initial state $s$, consider an initial state distributed according to some distribution $\rho \in \mathcal{M}(\mathcal{S})$.

The [[Policy Search Performance Measure|performance measure]] would be the average of following $\pi_{\theta}$ with the initial state $S_{1} \sim \rho$.
$$
J(\pi_{\theta})
= J_{\rho}(\pi_{\theta})
\triangleq \int V^{\pi_{\theta}}(s) \ d\rho(s)
= \mathbb{E}_{S \in \rho}[ \ V^{\pi_{\theta}}(S) \ ]
$$
where
- the [[Policy|optimal policy]] maximizes [[Value Function|cost function]] $J_{\rho}$.
- $J_{\rho}(\pi^{*}) \geq J_{\rho}(\pi_{\theta})$ for any [[Policy|policy]] $\pi_{\theta} \in \Pi_{\Theta}$.
- If $\pi^{*} \notin \Pi_{\Theta}$, the inequality is strict if the support of $\rho$ is the whole state space $S$.

In [[Policy-Based Methods|policy search methods]], we aim to find the maximizer of the [[Policy Search Performance Measure|performance measure]] within $\Pi_{\Theta}$.
$$
\tilde{\pi} \leftarrow 
\arg\max_{\pi_{\theta} \in \Pi_{\Theta}} 
J_{\rho}(\pi_{\theta})
$$
- The corresponding [[Policy|policy]] is identified by its parameter $\tilde{\theta}$: $\tilde{\pi} = \pi_{\tilde{\theta}}$.
- For different distributions $\rho$, we may get different optimizers.
- Hence to emphasize the dependance of the maximizer on $\rho$, we may use $\tilde{\pi}_{\rho}$.
- We may denote $J(\pi_{\theta})$ or $J_{\rho}(\pi_{\theta})$ simply by $J_{\rho}(\theta)$.

---
## Optimization problem
We need to solve the optimization problem $\tilde{\pi} \leftarrow \arg\max_{\pi_{\theta} \in \Pi_{\Theta}} J_{\rho}(\pi_{\theta})$ to find $\pi_{\theta}$ that maximizes the [[Policy Search Performance Measure|performance measure]] $J_{\rho}$.

- **Challenge**: The value $J_{\rho}$ is not readily available.
  Has to be estimated through interaction with environment.
- **Opportunity**: Special structure of the [[Markov Decision Process (MDP)|RL problem]].
  Such as the [[Value Function|value function]] satisfying the [[Bellman Equation for Value Function|Bellman equation]].

Optimization methods can be categorized based on the information they need about their objectives.
- **Zero-order methods** only use value of objective at various query points.
  They compute $J_{\rho}(\theta)$ at various $\theta$ in order to guide the optimization process.
- **First-order methods** use the derivative the objective instead or in addition to the value.
  They use $\Delta_{\theta} \ J_{\rho}(\theta)$ in order to guide the search.
  The quintessential first-order optimization method is [[Gradient Descent|gradient descent]].
