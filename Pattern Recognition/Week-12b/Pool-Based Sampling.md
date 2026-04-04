# Pool-Based Sampling
**Problem Setting**: The agent has a model $h_{t}$ trained from 
- a labelled dataset $S_{t}=((x_{1}, y_{1}), \ \dots, \ (x_{n}, y_{n}))$.
- a collection of unlabelled data point $U_{t} = (x_{n+1}, \dots, x_{n+m})$.
- and a budget $T \leq \infty$.

At each iteration, the agent must specify some number of queries for the oracle to label.

**(Greedy) Algorithm**: At each iteration, specify $k$ samples to label. Using the [[Acquisition Functions|acquisition function]], greedily select $x_{1}, \dots, x_{k} \in U_{t}$ such that
$$
\text{acq}(x_{1}, h_{t}, S_{t})
< \text{acq}(x_{2}, h_{t}, S_{t})
< \dots
< \text{acq}(x_{k}, h_{t}, S_{t})
$$
- Get labels from the oracle
- Update $S_{t+1} = S_{t} \cup \{ (x_{1}, y_{1}), \ \dots , \ (x_{k}, y_{k}) \}$
- Update $h_{t+1} = A(S_{t+1})$

**Optional**: Update $U_{t}$ to have more/fewer/different samples.
**Repeat** until $t\geq T$.

---
