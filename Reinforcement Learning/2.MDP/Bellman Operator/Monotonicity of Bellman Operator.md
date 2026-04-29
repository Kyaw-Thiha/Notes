# Monotonicity
#rl/bellman-equation/operator/monotonicity 
If $V_{1}(s) \leq V_{2}(s) \ , \forall s \in \mathcal{S}$ and we apply the [[Bellman Operator]] to both sides, we get
$$
T^{\pi}V_{1} \leq T^{\pi} V_{2}
$$
> The [[Bellman Operator]] does not change the order relationship.

![Monotonicity|400](https://notes-media.kthiha.com/Monotonicity-of-Bellman-Operator/85f4b933bb7913e67dca8239217d73a6.png)

---
## Theorem
**Lemma**
Let policy be denoted $\pi$.
If $V_{1}, V_{2} \in \mathcal{B}(\mathcal{S})$ and $V_{1} \leq V_{2}$, then we have
$$
\begin{align}
T^{\pi} V_{1} &\leq T^{\pi} V_{2} \\[6pt]
T^{*} V_{1} &\leq T^{*} V_{2}
\end{align}
$$

**Proof**: Let's expand $T^{\pi}V_{1}$.
As $V_{1}(s) \leq V_{2}(s)$ for any $s \in \mathcal{S}$,
$$
\begin{align}
(T^{\pi}V_{1})(s)
&= r^{\pi}(s) + \gamma \int \mathcal{P}^{\pi}
(ds' \mid s) \ \underbrace{V_{1}(s')} 
_{\leq V_{2}(s')} \\[6pt]

&\leq r^{\pi}(s) + \gamma \int \mathcal{P}^{\pi}
(ds' \mid s) V_{2}(s')
= (T^{\pi} V_{2})(s)
\end{align}
$$
$\therefore T^{\pi}V_{1} \leq T^{\pi}V_{2}$.

Likewise for [[Bellman Optimality Operator]],
$$
\begin{align}
(T^{*}V_{1})(s)
&= \max_{a \in \mathcal{A}} \left\{  
r(s,a) + \gamma \int \mathcal{P}(ds' \mid s,a) 
\underbrace{V_{1}(s')}_{\leq V_{2}(s')} \right\}
\\[6pt]
&\leq \max_{a \in \mathcal{A}} \left\{  
r(s,a) + \gamma \int \mathcal{P}(ds' \mid s,a) 
V_{2}(s') \right\}
= (T^{*}V_{2})(s)
\end{align}
$$
$\therefore T^{*}V_{1} \leq T^{*}V_{2}$.

---
## See Also
- [[Bellman Operator]]
- [[Bellman Equation for Value Function]]
- [[Bellman Equation for Optimal Value Functions]]
