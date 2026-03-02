# Greedy Policy
#rl/policy/greedy 

`Greedy policy` always selects the action with the highest [[Reward|return]] for current state to maximize immediate return.

---
## Optimal Policy from optimal value function
We have proven that if we know [[Bellman Equation for Optimal Value Functions|optimal value function]] $V^{*}$ and [[Bellman Equation for Optimal Quality Functions|optimal quality function]] $Q^{*}$, we can find the `optimal policy` $\pi$.

Hence for any $s \in \mathcal{S}$, the optimal policy is
$$
\begin{align}
\pi^{*}(s)
&= \arg\max_{a \in \mathcal{A}} Q^{*}(s,a) \\[6pt]
&= \arg\max_{a \in \mathcal{A}}
\left\{  r(s,a) + \gamma \int \mathcal{P} 
(ds' \mid s,a) \ V^{*}(s') \right\}
\end{align}
$$
We can interpret this as the following:
- It needs to act optimally in current and future time steps.
- Suppose the agent is going to act optimally in the future. 
  Then when it gets to next state $S' \sim \mathcal{P}(\cdot \mid s,a)$, 
	- it follows the optimal policy $\pi^{*}$.
	- the value of following $\pi^{*}$ is $V^{*}(S')$
- The expected value of acting optimally in the future is

$$
\int \mathcal{P}(ds' \mid s,a) \ V^{*}(s')
$$

- The performance of agent at current state $s$ is going to be

$$
r(s,a) + \gamma \int \mathcal{P}(ds' \mid s,a) 
\ V^{*}(s')
$$
- To act optimally now, the agent should chooses the action that maximizes this value.

---
## Greedy Policy
