# Q-Learning
[[Q-Learning]] can be thought of as a [[Stochastic Approximation(SA)|stochastic approximation]] of the [[Quality Function|Q-function]] by following the [[Greedy Policy|greedy policy]] using the current estimate.

$$
Q_{t+1}(S_{t}, A_{t})
\leftarrow (1 - \alpha_{t}(S_{t}, A_{t}))
\ \underbrace{Q_{t}(S_{t}, A_{t})}_{\text{Current Estimate}}
+ \alpha_{t}(S_{t}, A_{t}) \underbrace{
\left[ R_{t} + \gamma \max_{a' \in \mathcal{A}}
Q_{t}(S_{t+1}, a') \right]}_{\text{Return based on Greedy Policy}}
$$

---
## Main Idea
We shall apply [[Temporal Difference Learning for Policy Evaluation(TD)|TD-like]] methods for the problem of control.
Let's devise a sample-based asynchronous version of [[Value Iteration]] for the control problem.

## Derivation
Consider an arbitrary [[Quality Function|quality function]] $Q \in \mathcal{B}(\mathcal{S} \times \mathcal{A})$.
Let $S' \sim \mathcal{P}(\cdot \mid S,A)$ and $R \sim \mathcal{R}(\cdot \mid S,A)$ and define
$$
Y = R + \gamma \max_{a' \in \mathcal{A}}
Q(S', a')
$$
We then have 
$$
\begin{align}
\mathbb{E}[Y \mid S=s, A=a]
&= r(s,a) + \gamma \int \mathcal{P}(ds' \mid s,a)
\max_{a' \in \mathcal{A}} Q(s', a') \\[6pt]
&= (T^{*}Q)(s,a)
\end{align}
$$
So, $Y$ is an unbiased version of $(T^{*}Q)(s,a)$.
The [[Bellman Optimality Operator|empirical Bellman optimality operator]] is 
$$
(\hat{T}^{*}Q)(s,a) 
\triangleq R + \gamma \max_{a' \in \mathcal{A}}
Q(S', a')
$$

### Empirical Value Iteration
We can define a noisy version of [[Value Iteration|Value Iteration(Control)]] as
$$
Q_{k+1} \leftarrow \hat{T}^{*}Q_{k}
$$
similar to [[Temporal Difference Learning for Policy Evaluation(TD)|empirical value iteration in TD Learning for PI]], we can rewrite this as
$$
Q_{k+1}
\leftarrow \hat{T}^{*}Q_{k}
= T^{*} Q_{k}
+ \underbrace{\left( \hat{T}^{*}Q_{k} 
- T^{*}Q_{k} \right)}_{\triangleq \epsilon_{k}}
$$
where $\epsilon_{k}$ is the noise term $w.r.t$ optimal $T^{*}Q_{k}$.

### Stochastic Approximation
In order to diminish the effect noise, let's use a [[Archi Notes|SA procedure]].

For an observed $(S_{t}, A_{t})$, 
$$
Q_{t+1}(S_{t}, A_{t})
\leftarrow (1 - \alpha_{t}(S_{t}, A_{t}))
\ Q_{t}(S_{t}, A_{t})
+ \alpha_{t}(S_{t}, A_{t})
\left[ R_{t} + \gamma \max_{a' \in \mathcal{A}}
Q_{t}(S_{t+1}, a') \right]
$$
and for all other states $(s,a) \neq (S_{t}, A_{t})$ not being updated,
$$
Q_{t+1}(S_{t}, A_{t}) \leftarrow Q_{t}(s,a)
$$
where
- $Q_{t}(s,a)$ is the current estimate of the [[Quality Function|Q-function]].
- $Q_{t+1}(s,a)$ is the updated estimate of the [[Quality Function|Q-function]].
- $R_{t} + \gamma\max_{a' \in \mathcal{A}} Q_{t}(S_{t+1}, a')$ is the  [[Greedy Policy|greedy]] [[Reward|return]] of following the current estimate of [[Quality Function|Q-function]].
- $\alpha(S_{t}, A_{t})$ is the [[Learning Rate|learning rate]] $w.r.t$ $S_{t}$ and $A_{t}$.
  Note that this can sometimes be state-action independent $\alpha_{t}$ or even time-independent $a$.

### Greedy Policy
Looking at the update rule, we have $\max_{a' \in \mathcal{A}} Q_{t}(S_{t+1}, a')$.
Hence the policy being evaluated at $S_{t+1}$ is
$$
\begin{align}
\max_{a' \in \mathcal{A}} Q_{t}(S_{t+1}, a')
&= Q_{t} \left( S_{t+1}, \arg\max_{a' \in \mathcal{A}}
Q_{t}(S_{t}, a') \right) \\[6pt]
&= Q_{t}(S_{t+1}, \pi_{g}(S_{t}, Q_{t}))
\end{align}
$$
So, `Q-learning` is evaluating the [[Greedy Policy]] $w.r.t$ the current estimate of $Q_{t}$.

### Off-Policy Algorithm
Note that the [[Greedy Policy]] $\pi_{g}$ can be different from the [[Policy|policy]] $\pi$ that the algorithm follows.
Hence, `Q-learning` works under off-policy sampling scenario.

---
