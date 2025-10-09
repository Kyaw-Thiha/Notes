# Epsilon Greedy Strategy

Set initial $\epsilon=1.0$
`Exploitation`: $\text{probability }1-\epsilon$  (action with highest reward)
`Exploration`: $\text{probability }\epsilon$ (random action)

![[Epsilon Greedy Strategy.png]]

Since value of $\epsilon$ is high at the start, the agent will explore more in the beginning.
As the training goes on, value of $\epsilon$ is reduced and the agent more on focusing the better quality `policy`/`value function`.

## See Also
- [[Greedy Strategy]]
