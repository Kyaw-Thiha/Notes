# Bellman Equation
#rl/bellman-equation/value-function 
By [[Bellman Equation for Value Function]] for $\pi^{*}$, we get that
$$
V^{\pi^{*}} = r^{\pi^{*}} + \gamma 
\ \mathcal{P}^{\pi^{*}} V^{\pi^{*}}
$$
Now, let's find an optimal equation $V^{*}$ that does not refer to $\pi^{*}$.

---
## Claims
- **Claim-1**
  For any $s \in \mathcal{S}$, there exists a $V^{*}$ that satisfies

$$
V^{*}(s)
= \max_{a \in \mathcal{A}} \left\{  r(s, a) 
+ \gamma \int \mathcal{P}(ds' \mid s, a) 
\ V^{*}(s') \right\}
$$
This is called `Bellman optimality equation` for value function.

- **Claim-2**
  $V^{*}$ is same as $V^{\pi^{*}}$ when $\pi$ is restricted to be within the space of stationary policies.
- **Claim-3**
  For `discounted continuing` [[Markov Decision Process (MDP)|MDPs]], we can always find a stationary policy that is optimal within the space of all stationary and non-stationary policies.

**TLDR**: $V^{*}$ exists and is equal to $V^{\pi^{*}}$.

---
## Bellman Optimality Equation
The `Bellman Optimality Equation` for the [[Value Function|value function]] is
$$
V^{*}(s)
= \max_{a \in \mathcal{A}} \left\{  r(s, a) 
+ \gamma \int \mathcal{P}(ds' \mid s, a) 
\ V^{*}(s') \right\}
$$

---
## See Also
- [[Value Function]]
- [[Bellman Equation for Value Function]]
- [[Markov Decision Process (MDP)]]