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
## Sample Average
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
If $n\to \infty$, the estimate converges to $V^{\pi}$.
With finite samples, the behaviour of the estimate is 
$$
\hat{V}^{\pi}(s)
\approx V^{\pi}(s) + O_{P}\left( \frac{1}{\sqrt{ n }} \right)
$$
We can also use the [[Stochastic Approximation(SA)|stochastic approximation]] too.

---
## Monte Carlo Estimation (Init-State Only)
The following is a variation of the above idea
where the initial state $S_{1}$ at episode $i$ is selected randomly according to the distribution $\rho \in \mathcal{M}(\mathcal{S})$.

Require: Step-size schedule $(\alpha_{t}(s))_{t\geq1}$ for all $s \in \mathcal{S}$.
- Initialize $\hat{V}^{\pi}_{1}:\mathcal{S}\to \mathbb{R}$ arbitrarily $(e.g. \hat{V}^{\pi}_{1} = 0)$
- **for** each episode, do
	- Initialize $S_{1}^{(t)} \sim \rho$
	- **for** each step $k$ of the episode $t$, do
		- Follow $\pi$ to obtain $S_{1}^{(t)}, A_{1}^{(t)}, R_{1}^{t}, \ S_{2}^{(t)}, A_{2}^{(t)}, R_{2}^{t},$.
	- **end for**
	- Compute $G^{\pi(t)}_{1} = \sum_{k\geq1} \gamma^{k-1} R^{(1)}_{k}$	
	- Update

$$
\hat{V}^{\pi}_{t+1}(S^{(t)}_{1})
\leftarrow \left( 1 - \alpha_{t}(S_{1}^{(t)}) \right)
\ \hat{V}^{\pi}_{t}(S^{(t)}_{1})
+ \alpha_{t}(S^{(t)}_{1}) \ G^{\pi(t)}_{1}
$$
- **end for**

For [[Stochastic Approximation(SA)|SA]] to converge, we need to choose learning rate $(\alpha_{t}(s))_{t}$ $s.t.$
$$
\sum ^{\infty}_{t=0} \alpha_{t}(s) = \infty
\ \quad \ 
\sum ^{\infty}_{t=0} \alpha_{t}^{2}(s) < \infty
$$
Set a counter of
$$
n_{t}(s) \triangleq
|\{ S^{(i)}_{t} = s: 1 \leq i \leq t \}|
$$
then we can define
$$
\alpha_{t}(s) = \frac{1}{n_{t}(s)}
$$

---
## First-Visit and Every-Visit Monte-Carlo Estimators
- The Monte-Carlo Estimation might be wasteful on data.
- We go through many states $(S_{1}, \ S_{2}, \ S_{3}, \ , \dots)$ within an [[Episode|episode]] but only update first state estimate
- MC does not benefit from the recursive structure of the return and [[Value Function|value function]].

---
