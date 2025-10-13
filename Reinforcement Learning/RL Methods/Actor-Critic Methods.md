# Actor-Critic Methods

The main key idea is to combine [[Value-Based Methods]] & [[Policy-Based Methods]] by using different `policy` & `value functions`.

- `Actor` is a `policy function` $\pi_{\theta}(s)$
- `Critic` is a `value function` $q_{w}(s, a)$

![Actor-Critic Methods](https://media.geeksforgeeks.org/wp-content/uploads/20250224190459513673/Actor-Critic-Method.webp)

- Given current state $s_{t}$, the `Actor` $\pi_{\theta}(s)$ samples an action $a_{t} \sim \pi_{\theta}(a | s_{t})$.
- Given action $a_{t}$, the `Environment` produces the reward  and the next state $s_{t+1}$.
- The `Critic` estimates how good the action was using either $V_{w}(s_{t})$ or $Q_{w}(s_{t}, a_{t})$.
- Compute the `Advantage`
  $\delta_{t} = r_{t} + \gamma.Q_{w}(s_{t+1}, a_{t+1}) - Q_{w}(s_{t}, a_{t})$
- The `Actor` $\pi_{\theta}(a|s_{t})$ updates its parameters using the value from `Critic`
  Non-Advantage: $\triangle\theta = \alpha\nabla_{\theta}.(\log \pi_{\theta} (s_{t}|a_{t})) . Q_{w}(s_{t}, a_{t})$
  Advantage: $\triangle\theta = \alpha.\nabla_{\theta}(\log \pi_{\theta} (s_{t}|a_{t})) . \delta_{t}$
- The `Critic` updates its parameters using the `Advantage`
  $\triangle w = \beta. \nabla_{w}(Q_{w}(s_{t}, a_{t})).\delta_{t}$

## Advantage Function
Instead of directly using the `Action value function` $Q_{w}(s_{t}, a_{t})$, we can use the `Advantage Function` to further stabilize the training.

$$
A(s_{t}, a_{t}) = r_{t} + \gamma.Q(s_{t+1}, a_{t+1}) - Q(s_{t}, a_{t})
$$

## See Also
- [[Value-Based Methods]]
- [[Policy-Based Methods]]
