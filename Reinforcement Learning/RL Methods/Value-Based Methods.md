# Value-Based Methods
#rl/value

Instead of learning a `policy function`, we learn a `value function` that maps a state to expected value of being at that space.

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

## State Value Function
For each state, the `state value function` outputs the **expected return** if the agent starts at that state & follows the policy for all timesteps.

$$
V_{\pi}(s) = E_{\pi}[G_{t} | S_{t} = s]
$$

![[State Value Function.png]]

## Action Value Function
For each state & action pair, the `action-value function` outputs the **expected return** if the agent starts in that state, takes the action, and follows the policy forever after.

$$
Q_{\pi}(s, a) = E_{\pi}[G_{t} | S_{t} = s,\ A_{t} = t]
$$

![[Action Value Function.png]]