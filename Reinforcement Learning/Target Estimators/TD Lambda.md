# TD(λ)
#rl/td-lambda

`TD(λ)` combines the strengths of [[Monte Carlo]] and [[Temporal Difference]] by using *multiple-step lookahead updates* weighted by a **decay parameter** `λ`.

![[TD Lambda.png]]

## Intuition
Instead of updating the value function using only the immediate next state (as in one-step TD), `TD(λ)` averages over all *n-step returns*, blending short-term TD and long-term Monte Carlo information.

- When `λ = 0` → becomes **TD(0)** (pure 1-step TD).
- When `λ = 1` → becomes **Monte Carlo** (full-episode return).
- For `0 < λ < 1`, the update is a **weighted mix** of n-step TD targets:
  
  $$
  G_t^{(λ)} = (1 - λ) \sum_{n=1}^{∞} λ^{n-1} G_t^{(n)}
  $$



## Update Rule
The general TD(λ) update for state values:

$$
V(S_t) \leftarrow V(S_t) + \alpha [G_t^{(λ)} - V(S_t)]
$$

In the **online (backward view)** using *eligibility traces*:

$$
E_t(S) = γλE_{t-1}(S) + 𝟙(S = S_t)
$$

$$
V(S) \leftarrow V(S) + α [R_{t+1} + γV(S_{t+1}) - V(S_t)] \, E_t(S)
$$

Here:
- $E_t(S)$ tracks how recently and frequently a state was visited.
- $λ$ controls how long past states remain “eligible” for updates.

---

# See Also
- [[Temporal Difference]]
- [[Monte Carlo]]
