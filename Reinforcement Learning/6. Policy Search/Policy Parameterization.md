# Policy Parameterization
#rl/policy-methods 
Consider a [[Policy|stochastic policy]] $\pi_{\theta}: \mathcal{S} \to \mathcal{M}(A)$ parameterized by $\theta \in \Theta$.
The set $\Theta \subset \mathbb{R}^{p}$ is the parameter space.
The space of all parameterized policies is given by:
$$
\Pi_{\Theta} = \{ \pi_{\theta}: \mathcal{S} 
\to \mathcal{M}(\mathcal{A}): \theta \in \Theta \}
$$

---
## Examples
There are many choices as to how we can parameterize [[Policy|policy]] $\pi_{\theta}$
Consider a generic example based on [[Softmax Function|Boltzmann(or softmax) distribution]].

Given a function $f_{\theta}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ (eg: [[Neural Network]] or [[Decision Tree]] parameterized by $\theta$), the density of choosing action $a$ at state $s$ is
$$
\pi_{\theta}(a \mid s)
= \frac{\exp( \ f_{\theta}(s,a) \ )}
{\int \exp( \ f_{\theta}(s,a') \ ) \ da'}
$$
A special case would be $f_{\theta}(s,a) = \phi(s,a)^{T} \theta$ for some features $\phi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^{p}$ and $\theta \in \mathbb{R}^{p}$:
$$
\pi_{\theta}(a \mid s)
= \frac{\exp( \ \phi(s,a)^{T} \theta \ )}
{\int \exp(s, a')^{T} \theta \ da'}
$$
**Discrete Action Space**
When the action space $\mathcal{A}$ is discrete, $\pi_{\theta}(a \mid s)$ denotes the probability of choosing action $a$ at state $s$ instead of its density: 
$$
\pi_{\theta}(a \mid s)
= \frac{\exp( \ \phi(s,a)^{T}\theta \ )}
{\sum_{a' \in \mathcal{A}} 
\exp( \ \phi(s, a')^{T} \theta \ )}
$$

**Normal Distribution Action Space**
Let $\pi_{\theta}(\cdot \mid s)$ defines a normal distribution over action space with $\theta$ parameterizing its mean and covariance:
$$
\pi_{\theta}(\cdot \mid s)
= \mathcal{N}(\mu_{\theta}(s), \ \Sigma_{\theta}(s))
$$
If the action-space is $d_{\mathcal{A}}\text{-dimensional}$:
- Mean: $\mu_{\theta}: \mathcal{S} \to \mathbb{R}^{d_{\mathcal{A}}}$ 
- Covariance: $\Sigma_{\theta}: \mathcal{S} \to \mathcal{X}^{d_{\mathcal{A}}}_{+}$
  where $\mathcal{S}^{d_{\mathcal{A}}}_{+}$ refers to the set of $d_{\mathcal{A}} \times d_{\mathcal{A}}$ positive semi-definite matrices

---
## Ease of Work in Continuous Action Space
Explicit parameterization of [[Policy|policy]] allows us to easily choose a continuous action

For [[Value-Based Methods]], this can be challenging.
Even if we know $Q^{*}$, computing this optimal policy
$$
\pi^{*}(s)
= \pi_{g}(s; \mid Q^{*})
= \arg\max_{a \in \mathcal{A}} Q^{*}(s,a)
$$
requires an optimization problem in $\mathcal{A}$.
- Challenging if $\mathcal{A}$ is a high-dimensional space.
- [[Value Iteration]] and [[Policy Iteration]] requires repeated calculation of the [[Greedy Policy]]

---
