# Membership Query Synthesis
#ml/active-learning/membership-query-synthesis
**Problem Setting**: The agent has a model $h$ learned from a labelled dataset, $S = ( (x_{1}, y_{1}) \ , \dots, \ (x_{n}, y_{n}) )$ and a domain it can sample $x \in \mathcal{X}$.

**Algorithm**: For finite budget $T\leq \infty$ samples, the agent asks the oracle to label
$$
x_{t}^{*} = \arg\max_{x \in \mathcal{X}}
\ \text{acquisition}(x, h_{t}, S_{t})
$$
The agent observes the label:
$$
y_{t}^{*} = \text{oracle}(x^{*}_{t})
$$
With that observation, it updates its training set $S_{t+1} = S_{t} \cup \{ (x^{*}_{t}, y^{*}_{t}) \}$, and produces a new model:
$$
h_{t+1} = A(S_{t+1})
$$
**Repeat** this until $t=T$.

---