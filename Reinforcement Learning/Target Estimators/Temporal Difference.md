# Temporal Difference
#rl/temporal-difference

`Temporal Difference` waits only for one step $S_{t+1}$ to form a `TD Target` to update the `policy`/`value function` $V(S_{t})$.

Since we don't have access to expected return $G_{t}$, we estimate it by adding `current reward` $R_{t+1}$ and `discounted next state value` $\gamma.V(S_{t+1})$.

$$
V(S_{t+1}) = V(S_{t}) + \alpha [R_{t+1} + \gamma.V(S_{t+1}) - V(S_{t})]
$$

![[Temporal Difference.png]]

## Monte Carlo vs Temporal Difference
`Monte Carlo`: $V(S_{t}) = V(S_{t}) + \alpha.[G_{t} - V(S_{t})]$
`TD Learning`: $V(S_{t}) = V(S_{t}) + \alpha.[R_{t+1} + \gamma.V(S_{t+1}) - V(S_{t})]$

# See Also
- [[Monte Carlo]]
