# SARSA
#rl/algorithms/sarsa

`SARSA` is an [[On-Policy Learning|On-Policy]] [[Value-Based Methods|Value-Based]] method that uses a [[Temporal Difference]] approach train its `action-value function`.

## Training SARSA

1. We initialize the `Q-Table` for each state-action pair.
2. Choose an action based on [[Epsilon Greedy Strategy]].
3. Based on `action` $A_{t}$, we get `reward` $R_{t+1}$ and `state` $S_{t+1}$
4. Since this is [[Temporal Difference]] learning, we update $Q(S_{t}, A_{t})$ every step.

We calculate the `TD Target` using [[Epsilon Greedy Strategy]]
$$
y_{t} = R_{t+1} + \gamma.max_{a}Q(S_{t+1}, a)
$$

Then, we use that `TD-Target` to update the policy function.

$$
\begin{align}
Q(S_{t}, A_{t})  
&= Q(S_{t}, A_{t}) + \alpha.[y_{t} - Q(S_{t}, A_{t})] \\[6pt]
&= Q(S_{t}, A_{t}) + \alpha.[R_{t+1} + \gamma.max_{a}Q(S_{t+1}, a) - Q(S_{t}, A_{t})] \\
\end{align}
$$

## See Also
- [[On-Policy Learning]]
- [[Value-Based Methods]]
- [[Temporal Difference]]
- [[Q-Learning]]

