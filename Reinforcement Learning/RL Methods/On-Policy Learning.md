# On-Policy Learning
#rl/on-policy

`On-Policy Learning` means the agent learns about the `same policy` it uses to act.

The `behavior policy` (inference) and the `target policy` (training) are identical.

## Example
In **SARSA**, both acting and updating use the same ε-greedy policy:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
$$

---
# See Also
- [[Off-Policy Learning]]
- [[SARSA]]
- [[Q-Learning]]

