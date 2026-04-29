# Policy Gradient for Immediate Reward

---
## Problem Formulation
Suppose that we want to find a [[policy]] $\pi_{\theta}: \mathcal{S} \to \mathcal{M(A)}$ with $\theta \in \mathbb{R}^{p}$ that maximizes the [[Policy Search Performance Measure|performance]] for the immediate reward problem.

Recall that the [[Markov Decision Process (MDP)|interaction protocol]] is
- at episode $t$, $S_{t} \sim \rho \sim \mathcal{M(S)}$
- the agent chooses action $A_{t} \sim \pi_{\theta}(\cdot \mid S_{t})$.
- the agent receives reward $R_{t} \sim \mathcal{R}(\cdot \mid S_{t}, A_{t})$.
- the agent starts the new independent episode $t+1$.

In RL setting, state distribution $\rho$ and [[reward]] $\mathcal{R}$ are not directly available, but only found through samples.

---
## Performance Measure
The [[Policy Search Performance Measure|performance measure]] is
$$
\begin{align}
J_{\rho}(\pi_{\theta})
&= \int V^{\pi_{\theta}}(s) \ d\rho(s) \\[6pt]
&= \int r^{\pi_{\theta}}(s) \ d\rho(s) \\[6pt]
&= \int r(s,a) \ \pi_{\theta}(a \mid s) \ d\rho(s) da
\end{align}
$$
since the [[value function]] $V^{\pi_{\theta}}$ for the immediate reward problem is same as $r^{\pi_{\theta}}$.

Here, we consider the action space to be continuous and we assume that $\pi_{\theta}(\cdot \mid s)$ provides a density over the state space.

---
## Policy Gradient
The [[Gradient Descent|gradient]] of $J_{\rho}(\pi_{\theta})$ $w.r.t$ $\theta$ is
$$
\begin{align}
\nabla_{\theta} J_{\rho}(\pi_{\theta})
&= \int r(s,a) \nabla_{\theta} \pi_{\theta}
(a \mid s) \ d \rho(s) \ da \\[6pt]
&= \int d\rho(s) \int r(s,a)  
\nabla_{\theta} \pi_{\theta}(a \mid s) \ da \\[6pt]
&= \mathbb{E}_{S \sim \rho} \left[ 
\int r(S, a) \nabla_{\theta} \pi_{\theta}(a\mid S)   \ da \right]
\end{align}
$$
For discrete action spaces, the inner integral becomes
$$
\sum_{a\in \mathcal{A}} r(s,a) \nabla_{\theta} 
\pi_{\theta}(a \mid s)
$$

We call $\nabla_{\theta} J_{\rho}(\pi_{\theta})$ as the [[Policy Gradient for Immediate Reward Problem|policy gradient]].

---
### Improving Performance Measure
If we compute [[Policy Gradient for Immediate Reward Problem|policy gradient]], we can update the [[Policy Parameterization|policy parameters]] using the gradient ascent method:
$$
\theta_{k+1}
\leftarrow \theta_{k} + \alpha_{k} \nabla_{\theta}
J_{\rho}(\pi_{\theta_{k}})
$$
This is similar to [[Finite Difference Approximation|finite difference approximation]].

---
## Computing the Policy Gradient
### Known $\rho$ and $r$
$$
\nabla_{\theta} J_{\rho}(\pi_{\theta})
= \int r(s,a) \ \nabla_{\theta} \pi_{\theta}
(a \mid s) \ d\rho(s) \ da
$$
Assuming we know $\rho$ and $r$,
- for each $s \in \mathcal{S}$, compute the summation or integral over all $a \in \mathcal{A}$ of $r(s,a) \ \nabla_{\theta} \pi_{\theta}(a \mid s)$.
- weigh that term proportional to $\rho(s)$.
- take average over all $s$.

---
### Known $\rho$ and unknown $r$
Assume that $r$ is known, but $\rho$ can only be sampled.
We shall approximately solve this problem by sampling $S_{i} \sim \rho$ $(i = 1, \ \dots, \ n)$ and computing
$$
\frac{1}{n} \sum^{n}_{i=1}
\sum_{a \in \mathcal{A}} r(S_{i}, a)
\ \nabla_{\theta} \ \pi_{\theta}(a \mid S_{i})
$$
or
$$
\frac{1}{n} \sum^{n}_{i=1}
\int r(S_{i}, a)
\ \nabla_{\theta} \ \pi_{\theta}(a \mid S_{i})
$$

As $S_{i} \sim \rho$, this is an unbiased estimate of
$$
\nabla_{\theta} J_{\rho}(\pi_{\theta})
= \mathbb{E}_{S \sim \rho}
\left[ \sum_{a \sim \mathcal{A}} r(S, a) 
\ \nabla_{\theta} \pi_{\theta}(a \mid S) \right]
$$
or
$$
\nabla_{\theta} J_{\rho}(\pi_{\theta})
= \mathbb{E}_{S \sim \rho}
\left[ \int r(S, a) 
\ \nabla_{\theta} \pi_{\theta}(a \mid S) \right]
$$

---
### Unknown $r$ and $\rho$
The term 
$$
\sum_{a \in \mathcal{A}} r(s, a) 
\ \nabla_{\theta} \pi_{\theta}(a \mid s)
$$
can be interpreted as the expectation of
$$
r(s, A) \ \nabla_{\theta} \pi_{\theta}(A \mid s)
$$
when $A$ is coming from a uniform distribution with $q(a) = \frac{1}{|\mathcal{A}|}$.

We then have
$$
\begin{align}
&\sum_{a \in \mathcal{A}} r(s,a) \nabla_{\theta}
\pi_{\theta}(a \mid s) \\[6pt]
= &|\mathcal{A}| \ \sum_{a \in \mathcal{A}}
q(a) \ r(s,a) \ \nabla_{\theta} \pi_{\theta} 
(a \mid s)
\end{align}
$$

If the actions were coming from a uniform distribution, we could easily form an empirical estimate of these terms.
But the actions in the [[Markov Decision Process (MDP)|interaction protocol]] comes from distribution $\pi_{\theta}(\cdot \mid s)$, which in general is different distribution than a uniform one.

For this, we have two approach
- estimate $\hat{r} \sim r$ using data
- modify $r(s,A) \nabla_{\theta} \pi_{\theta}(A \mid s)$ to quantity that can be estimated from data

---
#### Modifying the function using calculus
**From Calculus**
For a function $f: \mathbb{R} \to \mathbb{R}$, we have
$$
\frac{d (\log f(x))}{dx}
= \frac{\frac{df}{dx}(x)}{f(x)}
$$
or more generally for a function $f: \mathbb{R}^{p} \to \mathbb{R}$, we have
$$
\nabla_{x} \log f(x)
= \frac{\nabla_{x} f(x)}{f(x)}
$$

**Applying it**
Using $\nabla_{x} f(x) = f(x)\nabla_{x} \log f(x)$, we get
$$
\begin{align}
\int r(s,a) \nabla_{\theta} \pi_{\theta}(a \mid s)
\ da
&= \int r(s,a) \nabla_{\theta} \log \pi_{\theta}  
(a \mid s) \ \pi_{\theta}(a \mid s) \quad da \\[6pt]
&= \mathbb{E}_{A \sim \pi_{\theta}(\cdot \mid s)}
[r(s, A) \ \nabla_{\theta} \log \pi_{\theta}(A \mid s)]
\end{align}
$$
with the desired quantity being written as expectation of
$$
r(s,A)\ \nabla_{\theta} \log \pi_{\theta}(A \mid s)
$$
where $A \sim \pi_{\theta}(\cdot \mid s)$.

Hence,
- We can estimate the gradient of the performance $w.r.t$ the [[Policy Parameterization|parameters of the policy]] using data available through the [[Markov Decision Process (MDP)|interaction]] of the agent with its environment.
- We use the following to update the [[Policy Parameterization|policy parameters]]:
$$
\theta_{k+1}
\leftarrow \theta_{k} + \alpha_{k} \nabla_{\theta}
J_{\rho}(\pi_{\theta_{k}})
$$
- Since this is unbiased but noisy estimate of the gradient, this becomes a [[Gradient Descent|stochastic gradient ascent]].

---

