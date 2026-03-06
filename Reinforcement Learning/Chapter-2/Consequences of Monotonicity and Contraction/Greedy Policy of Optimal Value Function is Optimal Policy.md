# Greedy Policy of Optimal Value Function is Optimal Policy
#rl/bellman-equation/operator 

> **Theorem**
> The value of the [[Greedy Policy]] of $V^{*}$ is $V^{*}$.
> We have $T^{\pi} V^{*} = T^{*} V^{*}$ if and only if $V^{\pi} = V^{*}$.

---
## Proof
**Direction-1**: $T^{\pi}V^{*} = T^{*}V^{*} \implies V^{\pi} = V^{*}$
Assume that $T^{\pi}V^{*} = T^{*}V^{*}$.
As $V^{*}$ is the solution of [[Bellman Optimality Operator|Bellman optimality equation]],  $T^{*} V^{*} = V^{*}$.
Therefore,
$$
T^{\pi} V^{*} = T^{*} V^{*} = V^{*}
$$
This shows that $V^{*}$ is a [[Fixed Point|fixed point]] of $T^{\pi}$.
By [[Uniqueness of Fixed Point|this theorem]], the fixed point of $T^{\pi}$ is unique and equal to $V^{\pi}$.
So, $V^{\pi}$ and $V^{*}$ should be the same $V^{\pi} = V^{*}$.


**Direction-2**: $V^{\pi} = V^{*} \implies T^{\pi} V^{*} = T^{*}V^{*}$
We apply $T^{\pi}$ to both sides of $V^{*} = V^{\pi}$ to get
$$
T^{\pi} V^{*} = T^{\pi} V^{\pi}
$$
As $V^{\pi}$ is the solution of the [[Bellman Equation for Value Function|Bellman equation]] for [[Markov Policy|policy]] $\pi$, we have $T^{\pi}V^{\pi}=V^{\pi}$.
Hence,
$$
T^{\pi} V^{*}
= T^{\pi} V^{\pi}
= V^{\pi}
$$
By assumption $V^{\pi} = V^{*}$, we have $T^{\pi}V^{*} = V^{\pi} = V^{*}$.
On the other hand, $V^{*} = T^{*} V^{*}$ so
$$
T^{\pi}V^{*} = V^{*} = T^{*}V^{*}
$$
as wanted.

---
## Discussion
- Suppose $T^{\pi}V^{*} = T^{*}V^{*}$ for some policy $\pi$.
  We now have that the [[Value Function|value function]] $V^{\pi}$ of that [[Markov Policy|policy]] is the same as the [[Fixed Point|fixed point]] of $T^{*}$, which is $V^{*}$.

- But we have not shown that the fixed point of $T^{*}$ is the [[Value Function|optimal value function]], as in

$$
\pi^{*} = \arg\max_{\pi \in \Pi} V^{\pi}(s) \ , \quad \forall s \in \mathcal{S}
$$

- But this can be proven true using this [[Fixed Point of T* is Optimal Value Function|theorem]].

### Connection to Greedy Policy
To better see the relation to [[Greedy Policy]], consider the following

- Given $V^{*}$, the greedy policy selects

$$
\pi_{g}(s, V^{*})
= \arg\max_{a \in \mathcal{A}} \left\{  r(s,a) 
+ \gamma \int \mathcal{P}(ds' \mid s,a) 
\ V^{*}(s')  \right\}
$$

- So, we get

$$
T^{\pi_{g}(V^{*})} \ V^{*}
= \arg\max_{a \in \mathcal{A}} \left\{  r(s,a) 
+ \gamma \int \mathcal{P}(ds' \mid s,a) 
\ V^{*}(s')  \right\}
$$

- Recall that for $T^{*}V^{*}$, we have

$$
(T^{*}V^{*})(s) = \max_{a \in \mathcal{A}}
\left\{  r(s,a) + \gamma \int \mathcal{P}
(ds' \mid s,a) \ V^{*}(s')  \right\}
$$

- Hence, we now conclude that $T^{\pi_{g}(V^{*}) \ V^{*}} = T^{*} V^{*}$ 

This proposition states that the value of following $\pi_{g}(V^{*})$ is the same as $V^{*}$.

Practically, this means if we find $V^{*}$ and its greedy policy $\pi_{g}(V^{*})$, it is the same as $V^{*}$.

> To find an [[Markov Policy|optimal policy]], we can find $V^{*}$ first and then follow its [[Greedy Policy|greedy policy]] $\pi_{g}(V^{*})$.

---
