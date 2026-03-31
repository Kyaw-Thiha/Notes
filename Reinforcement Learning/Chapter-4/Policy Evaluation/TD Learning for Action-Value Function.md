We can use similar procedure to [[Temporal Difference Learning for Policy Evaluation(TD)|TD Learning for PE]] in order to estimate the `action-value function`.

To evaluate $\pi$, we need to have an estimate of $(T^{\pi}Q)(s,a)$ for all $(s,a) \in \mathcal{S} \times \mathcal{A}$. Suppose that state-action pair $(S_{t}, A_{t}) \sim \mu$ and new state $S_{t}' \sim \mathcal{P}(\cdot \mid S_{t},A_{t})$ and reward $R_{t} \sim \mathcal{R}(\cdot \mid S_{t}, A_{t})$.

The update rule would then be
for $(s,a) = (S_{t}, \ A_{t})$,
$$
\boxed{ \ Q_{t+1}(S_{t}, A_{t}) \leftarrow 
Q_{t}(S_{t}, A_{t}) + \alpha_{t}(S_{t}, A_{t})
[ \ R_{t} + \gamma Q_{t}(S_{t}', \ \pi(S_{t}'))
- Q_{t}(S_{t},A_{t}) \ ] \ }
$$
and all other $(s,a) \neq (S_{t}, A_{t})$,
$$
\boxed{ \ Q_{t+1}(s,a) \leftarrow Q_{t}(s,a) \ }
$$

Its easy to see that 
$$
\mathbb{E}[ \ R_{t} + \gamma Q_{t}(S_{t}', 
\ \pi(S_{t}')) \mid S=s, A=a \ ]
= (T^{\pi}Q)(s,a)
$$

---
## On-Policy and Off-Policy Sampling Scenarios
Recall the update rule:
$$
Q_{t+1}(S_{t}, A_{t}) \leftarrow 
Q_{t}(S_{t}, A_{t}) + \alpha_{t}(S_{t}, A_{t})
[ \ R_{t} + \gamma Q_{t}(S_{t}', \ \pi(S_{t}'))
- Q_{t}(S_{t},A_{t}) \ ]
$$
Then, we can observe the following:
- $\pi$ appears only in $Q_{t}(S_{t}', \ \pi(S_{t}'))$ term.
- The action $A_{t}$ does not need to be selected by $\pi$ itself.

This entails that the agent can generate the stream of data
$$
S_{1}, A_{1}, R_{1}, \ S_{2}, A_{2}, R_{2}, \ \dots
$$
by following a `behavior policy` $\pi_{b}$ that is different from the [[Markov Policy|policy]] that we want to evaluate $\pi$.

> - When $\pi_{b} = \pi$, we are in `on-policy` sampling scenario.
>   The agent is evaluating the same policy that it is following.
> - When $\pi_{b} \neq \pi$, we are in `off-policy` sampling scenario.
>   The agent is evaluating a policy different from the one it is following.

---
## See Also
- [[Temporal Difference Learning for Policy Evaluation(TD)]]
- [[Temporal Difference(TD) Error]]
- [[Markov Policy]]