# Solving Reinforcement Learning
#rl/policy #rl/value

## Policy-Based Methods
The `policy` $\pi$ is the function that tells what `action` to takes, given the `state` the agent is in.

![[Policy-Based Learning.png]]

### Deterministic
`Deterministic`: A policy $\pi$ that that return the same `action`, given the same `state`
$$
\pi(s) = a
$$

### Stochastic
`Stochastic`: A policy $\pi$ that outputs a probability distribution over actions

$$
\pi(a | s) = P[A|s]
$$

### Examples
- REINFORCE
- PPO (Proximal Policy Optimization)
- TRPO (Trusted Region Policy Optimization)
- DDPG, SAC (for continuous)

[[Policy-Based Methods|Read More]]

---
## Value-Based Methods
Instead of learning a policy function, we learn a `value function` that maps a state to expected value of being at that space.

$$
v_{\pi} (s) = E_{\pi}[R_{t+1} + \gamma.R_{t+2} + \gamma^2.R_{t+3} + \dots | S_{t} = s]
$$

![[Value-Based Learning.png]]

This mean we are not training a `policy funciton`.
Instead, the `policy` is a just a predefined function (like Greedy Policy).
This `policy` function uses the values given by `value function`, to select its actions.

### Examples
- Q-Learning
- SARSA
- Deep Q-Network (DQN)

[[Value-Based Methods|Read More]]

---
## Actor-Critic Methods
`Actor-Critic` methods combine learning both [[#Policy-Based Methods]] and [[#Value-Based Methods]].

![[Actor-Critic Learning.png]]

`Actor`: learn the policy $\pi_{\theta}(a|s)$
`Critic`: learn the value function $V_w(s)$ or $Q_w(s,a)$

This allow the critic to reduce variance in the actor's policy gradient estimates.

### Examples
- A2C / A3C
- DDPG
- TD3
- SAC

[[Actor-Critic Methods|Read More]]

## The link between Value & Policy Function

In `policy-based training`, the optimal policy $\pi^*$ is found by training the policy directly.
In `value-based training`, finding optimal value function $Q^*$ leads to having an optimal policy.
$$
\pi^*(s) = argmax_{a} Q^*(s, a)
$$

