# Greedy Strategy
#rl/greedy-strategy

Set fixed $\epsilon = 0.0$

`Exploitation`: $\text{probability }1.0$ (always choose action with highest reward)  
`Exploration`: $\text{probability }0.0$ (no random actions)

Since $\epsilon$ is zero, the agent never explores and always picks the current best-known action.  
This can lead to `suboptimal policies` if the agent gets stuck in a local optimum early in training.

> Greedy strategy is often used after training to evaluate or deploy the learned policy.

---

# See Also
- [[Epsilon Greedy Strategy]]
