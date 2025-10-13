# Monte Carlo
#rl/monte-carlo

`Monte Carlo` training waits till the end of a training episode, before calculating $G_{t}$ (return) and using it as a target for updating `policy`/`value function` $V(S_{t})$.

$$
V(S_{t}) = V(S_{t}) + \alpha [G_{t} - V(S_{t})]
$$

![[Monte Carlo.png]]

1. Start the training episode at the same starting point.
2. The `agent` takes action based on the current `policy`.
3. We get `reward` and the `next state`.
4. At the end of the episode, we have a list of `State`, `Action`, `Reward` and `Next States`.
5. The agent will sum the total rewards $G_{t}$.
6. It will then use $G_{t}$ to update the $V(s_{t})$.

## Monte Carlo vs Temporal Difference
`Monte Carlo`: $V(S_{t}) = V(S_{t}) + \alpha.[G_{t} - V(S_{t})]$
`TD Learning`: $V(S_{t}) = V(S_{t}) + \alpha.[R_{t+1} + \gamma.V(S_{t+1}) - V(S_{t})]$

## Monte Carlo Sampling
In actual training with `Monte Carlo`, we would calculate $G_{t}$ for each states $S_{t}$ at the end of the episode. 
Then for each state, we would average out the $G_{t}$ to estimate $V(S_{t})$.

$$
V(s) \approx \frac{1}{N(s)} \sum^{N(s)}_{i=1} G_{t}^{(i)}
$$

Compared to [[Temporal Difference]], it's unbiased since we are not estimating the return.

However due to `stochasticity` of the environment & policy, different `trajectories` can lead to vastly different values.
Thus, it suffers from high variance.

One of the way to solve this is using large number of trajectories to smooth out the variance.
However, this leads to higher batch size.

So, a good approach to solve this is to use [[Actor-Critic Methods]].

## See Also
- [[Temporal Difference]]

