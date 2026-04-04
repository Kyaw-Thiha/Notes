# VFA given Value Function
#rl/vfa
**Main Idea:**
Suppose we happen to know $V^{\pi}$ $(\text{or } Q^{\pi}, V^{*}, Q^{*})$.
We want to represent it with a function $V \in \mathcal{F}$.
Our goal can then be formulated as
$$
V \approx V^{\pi}
$$

**Quantifying distance**
To quantify $V \approx V^{\pi}$, we have to pick a distance function between function $V$ and $V^{\pi}$, $d:\mathcal{B(S) \times B(S)} \to \mathbb{R}$.
We can then express our goal as
$$
V \leftarrow \arg\min_{V \in \mathcal{F}}
d(V, V^{\pi})
$$

**Norm**
We could use the [[p-Norm]] $w.r.t$ a probability measure $\mu \in \mathcal{M}(\mathcal{S})$ to define distances between functions by 
$$
d(V_{1}, V_{2}) = ||V_{1} - V_{2}||_{p,\mu}
$$
We then have
$$
V \leftarrow \arg\min_{V \in \mathcal{F}}
||V - V^{\pi}||_{p, \mu}
$$
A common choice would be $L_{2}\text{-norm}$.

---
## How do have access to Value Function?
**How do we have access?**
Given that we are trying to approximate the [[Value Function]], how would we even have access to it?
	- In [[Monte Carlo Estimation for Policy Evaluation|Monte-Carlo estimation]]: For a state $s$, we can have $V^{\pi}(s) + \epsilon(s)$ with $\mathbb{E}[\epsilon(s)] = 0$.

**Why even need approximation then?**
If [[Monte Carlo Estimation for Policy Evaluation|Monte Carlo estimate]] can give an estimate for the value of a state, what is the reason for using [[VFA given Value Function]] after all?
	- When state space is large $(\text{e.g: continuous})$, we cannot run [[Monte Carlo Estimation for Policy Evaluation|MC]] for all states, but only finite number of them.
	- [[Monte Carlo Estimation for Policy Evaluation|MC]] estimates are noisy.

Hence, the role of [[VFA given Value Function]] is to generalize from finite number of noisy data to the whole state space.

---
