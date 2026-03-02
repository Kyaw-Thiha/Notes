# Bellman Equation
#rl/bellman-equation/quality-function 
This note explore the relation between 
- optimal policy for [[Value Function|value function]] $\pi^{*}_{V}$
- and optimal policy for [[Quality Function|quality function]] $\pi^{*}_{V}$

Optimal state-value function is defined as
$$
\pi \geq \pi'
\iff
\forall s \in \mathcal{S}, \ V^{\pi}(s) \geq V^{\pi'}(s)
$$

Likewise, optimal action-value function is defined as
$$
\pi \geq \pi'
\iff
\forall \ (s,a) \in \mathcal{S} \times \mathcal{A}, 
\ Q^{\pi}(s, a) \geq Q^{\pi'}(s,a)
$$

---
## Bellman Optimality Equation for Q-Function
For any $(s,a) \in \mathcal{S} \times \mathcal{A}$,
$$
\boxed{ \
Q^{*}(s,a)
= r(s,a) + \gamma \int \mathcal{P}(ds' \mid s,a)
\max_{a' \in \mathcal{A}} Q^{*}(s', a')
\ }
$$

---
## Deriving the Bellman Equation
### Proposition-1
> **Proposition**: Given two policies $\pi, \ \pi'$, 
> - $V^{\pi} \geq V^{\pi'} \implies Q^{\pi} \geq Q^{\pi^{'}}$ 
> - and $V^{\pi} > V^{\pi'} \implies Q^{\pi} > Q^{\pi'}$.

**Proof**

Suppose that $\forall s \in \mathcal{S}, \ V^{\pi}(s) \geq V^{\pi'}(s)$.
Rewriting action-value functions $Q^{\pi}$ and $Q^{\pi^{*}}$ in terms of value function $V^{\pi}$ and $V^{\pi^{'}}$, we get that $\forall (s,a) \in \mathcal{S}$,
$$
\begin{align}
Q^{\pi}(s,a)
&= r(s,a) + \gamma \int \mathcal{P}(ds' \mid s,a)
\ V^{\pi}(ds') \\[6pt] 

&\geq r(s,a) + \gamma \int \mathcal{P}(ds' \mid s,a)
\ V^{\pi'}(ds')
= Q^{\pi'}(s,a)
\end{align}
$$
$\therefore V^{\pi} \geq V^{\pi'} \implies Q^{\pi} \geq Q^{\pi'}$.

The proof is analogous for strict inequality.
Note that support of $\mathcal{P}(\cdot \mid s,a)$ overlaps $S' = \{ s': V^{\pi}(s') > V^{\pi'}(s') \}$.

---
The relation above is only one-way.
$$
\begin{align}
&V^{\pi} \geq V^{\pi'} \implies Q^{\pi} \geq Q^{\pi'}
\\[6pt]
&Q^{\pi} \geq Q^{\pi'}  
\nRightarrow V^{\pi} \geq V^{\pi'}
\end{align}
$$
But for optimal policies $\pi^{*}_{V}$ and $\pi^{*}_{Q}$ with $V^{\pi^{*}} \geq V^{\pi'}$ and $Q^{\pi^{*}} \geq Q^{\pi'}$, we can show that they have same value functions.

---
### Proposition-2
> **Proposition**: Let $\pi^{*}_{V} \leftarrow \arg\max_{\pi \in \Pi}$ and $\pi^{*}_{Q} \leftarrow \arg\max_{\pi \in \Pi}$.
> We have $V^{\pi^{*}_{V}} = V^{\pi^{*}_{Q}}$.

**Proof**
We prove by first showing $V^{\pi^{*}_{V}} \geq V^{\pi^{*}_{Q}}$ and $V^{\pi^{*}_{V}} \leq V^{\pi^{*}_{Q}}$, and then
$$
\left( V^{\pi^{*}_{V}} \geq V^{\pi^{*}_{Q}} \right)
\ \land \ 
\left( V^{\pi^{*}_{V}} \leq V^{\pi^{*}_{Q}} \right)
\implies 
\left( V^{\pi^{*}_{V}} = V^{\pi^{*}_{Q}} \right)
$$

**Statement-(a)**: $\forall s \in \mathcal{S}, \ V^{\pi^{*}_{V}}(s) \geq V^{\pi^{*}_{Q}}(s)$.
Note that $\pi^{*}_{V}$ is the maximizer $\pi^{*}_{V} \leftarrow \arg\max_{\pi \in \Pi} V^{\pi}$.
Thus, $V^{\pi^{*}_{V}} = \max_{\pi \in \Pi} V^{\pi} \geq V^{\pi'}$ for any policy $\pi'$ including $\pi' = \pi^{*}_{Q}$.
$\therefore$ $\forall s \in \mathcal{S}, \ V^{\pi^{*}_{V}}(s) \geq V^{\pi^{*}_{Q}}(s)$, **as wanted**.

**Statement-(b)**: $\forall s \in \mathcal{S}, \ V^{\pi^{*}_{V}}(s) \leq V^{\pi^{*}_{Q}}(s)$.
By maximizer property of $\pi^{*}_{Q}$, we have $Q^{\pi^{*}_{Q}} = \max_{\pi \in \Pi} Q^{\pi} \geq Q^{\pi'}$ for any policy $\pi'$ including $\pi' = \pi^{*}_{V}$.
Therefore,
$$
Q^{\pi^{*}_{Q}}(s,a)
\geq Q^{\pi^{*}_{V}}(s,a)
\quad -(1)
$$
for all $(s,a) \in \mathcal{S} \times \mathcal{A}$.

To prove by **contradiction**, suppose $V^{\pi^{*}_{V}} > V^{\pi^{*}_{Q}}$.
Using [[#Proposition-1]], we get that 
$$
V^{\pi^{*}_{V}} > V^{\pi^{*}_{Q}}
\implies Q^{\pi^{*}_{V}} > Q^{\pi^{*}_{Q}}
\quad -(2)
$$
Under this assumption $\exists (s,a) \in \mathcal{S} \times \mathcal{A}$ $s.t.$ $Q^{\pi^{*}_{V}}(s,a) > Q^{\pi^{*}_{Q}}(s,a)$.

Using statements $(1)$ and $(2)$, we got that
$$
Q^{\pi^{*}_{Q}} (s,a)
\geq Q^{\pi^{*}_{V}}(s,a)
> Q^{\pi^{*}_{Q}}(s,a)
$$
Note that this is impossible.
Hence by contradiction, we get that $\neg \ \exists s \in \mathcal{S}, \ V^{\pi^{*}_{V}}(s) > V^{\pi^{*}_{V}}$.
$\therefore$ $\forall s \in \mathcal{S}, \ V^{\pi^{*}_{V}}(s) \leq V^{\pi^{*}_{Q}}(s)$, **as wanted**.

---
### Final Conclusion
Using **Statement-(a)** and **(b)**, we have proven that
$$
V^{\pi^{*}_{V}} = V^{\pi^{*}_{Q}}
$$
Hence, we can define the optimal [[Markov Policy|policy]] as 
- the policy that satisfies $\pi^{*} \leftarrow \arg \max_{\pi \in \Pi} Q^{\pi}$
- or the policy that satisfies $\pi^{*} \leftarrow \arg \max_{\pi \in \Pi} V^{\pi}$
- and they both have the same [[Value Function|value function]].

This allows us to define the Bellman optimality equation for the action-value functions.

For any $(s,a) \in \mathcal{S} \times \mathcal{A}$,
$$
Q^{*}(s,a)
= r(s,a) + \gamma \int \mathcal{P}(ds' \mid s,a)
\max_{a' \in \mathcal{A}} Q^{*}(s', a')
$$

We can also show that such a $Q^{*}$ exists and is unique.
Likewise, we have that $Q^{*} = Q^{\pi^{*}}$.
The proof for those can be found in [[Bellman Equation for Optimal Value Functions|optimality for value function]].

---
## See Also
- [[Bellman Equation for Optimal Value Functions]]
- [[Quality Function]]
- [[Value Function]]