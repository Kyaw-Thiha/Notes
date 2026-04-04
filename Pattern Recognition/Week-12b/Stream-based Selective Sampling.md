# Stream-based Selective Sampling
#ml/active-learning/stream-based-selective-sampling
**Problem Setting**: An agent has a model $h_{t}$.
This has been trained on a labelled dataset $S_{t}$ and a sampling budget $T \leq \infty$.

The agent is sequentially presented with inputs from an unlabelled dataset $x_{i} \in \mathcal{U}$. The can either ignore $x_{i}$, or ask the oracle to sample.

**(Naive) Algorithm**: For a given [[Acquisition Functions|acquisition function]], define a "region of uncertainty" $i.e.$
$$
R_{U} = \{ x: \text{acquisition}
(x, h_{t}, S_{t}) \geq \theta \}
$$
If $x_{i} \in R_{U}$, request oracle to label it. Otherwise, ignore it.
Then, $S_{t+1} = S_{t} \cup \{ (x_{i}, y_{i}) \}$ and $h_{t+1} = A(S_{t+1})$.
**Repeat** until $t \geq T$.

---