# Reinforcement Learning Setting
#rl
> In `RL setting`, we do not have access to the [[Markov Decision Process (MDP)|MDP model]].
> Instead, we observe data of agents interacting with the environment.

This is in contrast to the [[Planning (Reinforcement Learning)|Planning setting]] where the model ($\mathbf{\mathcal{P}}$ and $\mathcal{R}$) are known.

---
### Stream of Data
The `stream of data` can be represented as
$$
S_{1}, A_{1}, R_{1}, \ S_{2}, A_{2}, R_{2} , \ \dots
$$
with 
- $A_{t}\sim \pi(\cdot \mid S_{t})$
- $S_{t+1} \sim \mathcal{P}(\cdot \mid S_{t}, \ A_{t})$
- $R_{t} \sim \mathcal{R}(\cdot \mid S_{t}, \ A_{t})$

---
## See Also
- [[Planning (Reinforcement Learning)]]