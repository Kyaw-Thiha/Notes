# Online Learning of the Reward Function
Akin to [[Online Estimator of Mean of Random Variable]], we shall be using [[Stochastic Approximation(SA)|stochastic approximation]] to carry out online learning of the [[Reward|reward function]].

---
## Problem Setting
Recall the [[Markov Decision Process (MDP)|immediate reward process]] that at episode $t$,
- the agent starts at state $S_{t} \sim p \in \mathcal{M}(\mathcal{S})$
- chooses action $A_{t} \sim \pi(\cdot \mid S_{t})$
- and receives a reward of $R_{t} \sim \mathcal{R}(\cdot \mid S_{t}, A_{t})$ 
- the agents starts a new independent episode $t+1$ and the process repeats

When the [[Reward|reward function]] $r:\mathcal{S} \times \mathcal{A} \to \mathbb{R}$ was known, the [[Policy|optimal policy]] would be
$$
\pi^{*}(s) \leftarrow \arg\max_{a \in \mathcal{A}}
r(s,a)
$$

But what if we don't know the [[Reward|reward function]]?
We can use [[Stochastic Approximation(SA)|stochastic approximation]] to estimate $r(s,a)$.

---
## Online Learning
We shall use [[Stochastic Approximation(SA)|SA]] to estimate $r(s,a)$.
This would be an extension of how we [[Online Estimator of Mean of Random Variable|estimated the mean of a single variable]] $Z \sim v$.

In this case, we will be having **multiple** random variables, each for one of the state-action pairs $(s,a) \in \mathcal{S} \times \mathcal{A}$.
Each $r.v.$ has a distribution $\mathcal{R}(\cdot \mid s,a)$ with a mean of 
$$
r(s,a) = \mathbb{E}[R \mid S=s, A=a]
$$
with $R \sim \mathcal{R}(\cdot \mid s,a)$ for all $(s,a) \in \mathcal{S} \times \mathcal{A}$.

---
### Methodology
Let $\hat{r}_{t}:\mathcal{S} \times \mathcal{A} \to \mathbb{R}$ as our estimate of [[Reward|reward]] $r$ at time $t$.
Let state-action indexed sequence $\alpha_{t}(s,a)$ be the step size.
At time/episode $t$, the state-action pair $(S_{t}, A_{t})$ is selected.

We update $\hat{r}_{t}(S_{t}, A_{t})$ with [[Stochastic Approximation(SA)|stochastic approximation]]:
$$
\hat{r}_{t+1}(S_{t}, A_{t})
\leftarrow (1 - \alpha_{t}(S_{t}, \ A_{t})) 
\ \hat{r}_{t}(S_{t}, \ A_{t})
+ \alpha_{t}(S_{t}, A_{t}) \ R_{t}
$$
Note the [[Stochastic Approximation(SA)|SA form]] similar to our derivation in [[Online Estimator of Mean of Random Variable|online estimation of mean of random variable]].

For all $(s,a) \neq (S_{t}, A_{t})$, we do not change our estimate $\hat{r}_{t+1}(s,a)$ from what we had as $\hat{r}(s,a)$.
This can be denoted with $\alpha_{t}(s,a)=0$ as
$$
\hat{r}_{t+1}(s,a) \leftarrow (1 - 0) \
\hat{r}_{t}(s,a) + 0 \cdot R_{t}
$$

---
### Stochastic Approximation Conditions
The [[Stochastic Approximation(SA)|stochastic approximation]] conditions apply here
$$
\begin{align}
&\sum^{\infty}_{t=0} \alpha_{t}(s,a) = \infty \\[6pt]
&\sum^{\infty}_{t=0} \alpha_{t}^{2}(s,a) \neq \infty
\end{align}
$$
with the difference that it should be for each state-action pairs $(s,a) \in \mathcal{S} \times \mathcal{A}$.

---
### Defining the step size
To define $\alpha_{t}(s,a)$, we can use a counter on how many times $(s,a)$ has been picked up to time $t$.

Let's define the following:
$$
n_{t}(s,a) \triangleq |\{ i: (S_{i}, A_{i})
= (s,a) \ , \quad i=1, \ \dots, \ t \}|
$$
We can then choose $\alpha_{t}(s,a) = \frac{1}{n_{t}(s,a)}$.
This naturally leads to $\hat{r}_{t}(s,a)$ being a [[Sample Average Estimator|sample mean]] of all [[Reward|rewards]] encountered at $(s,a)$.

---
### Caveats
Now consider what would happen if 
- the sampling distribution $X_{t} \sim \rho$ never chooses state $x_{0}$
- the policy $\pi(\cdot \mid x_{0})$ never chooses a particular action $a_{0}$ at a certain state $s_{0}$

This is after all possible since we cannot have infinite number of samples from all state-actions pairs 
Hence, we cannot satisfy the condition $\sum^{\infty}_{t=0} \alpha_{t}(x_{0}, \ a_{0})=\infty$.

---
### Epsilon Greedy Strategy
We can use a certain chance of taking an action from a uniform distribution instead of following a [[Greedy Policy]].
This full-fill the condition of infinite number of samples, and is called [[Epsilon Greedy Policy]].

---
## See Also
- [[Online Estimator of Mean of Random Variable]]
- [[Stochastic Approximation(SA)]]
- [[Reward]]
- [[Greedy Policy]]
- [[Epsilon Greedy Policy]]