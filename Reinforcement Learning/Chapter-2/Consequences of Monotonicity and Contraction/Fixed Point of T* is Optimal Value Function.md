# $V^{*}$ is the same as $V^{\pi^{*}}$
#rl/bellman-equation/operator 

**Theorem**
Let $V^{*}$ be the [[Fixed Point]] of $T^{*}$, $i.e.$ $V^{*} = T^{*}V^{*}$.
We then have
$$
V^{*}(s) = \sup_{\pi \in \Pi} V^{\pi}(s) 
\ , \quad \forall s \in \mathcal{S}
$$
---
## Proof
We first show that $V^{*}(s) \leq \sup_{\pi \in \Pi} V^{\pi}(s)$, $\forall s \in \mathcal{S}$.
Then, we show the opposite direction $\sup_{\pi \in \Pi} V^{\pi}(s) \leq V^{*}(s)$.
We can then combine those two to prove the theorem.

**Direction-1**
Using the [[Error Bounds of Bellman Equation]] with the choice of $V = V^{*}$, we get that for any $\pi \in \Pi$,

$$
||V^{*} - V^{\pi}||_{\infty}
\leq \frac{||V^{*} - T^{\pi}V^{*}||_{\infty}}{1- \gamma}
$$
For $\epsilon > 0$, choose a [[Markov Policy|policy]] $\pi_{\epsilon} \in \Pi$ such that
$$
||V^{*} - T^{\pi_{\epsilon}}V^{*}||_{\infty}
\leq (1 - \gamma) \ \epsilon
$$

This is possible because by definition of [[Bellman Operator]], we have
$$
(T^{*}V^{*})(s)
= \sup_{a \in \mathcal{A}} \left\{  r(s,a) + \gamma \ \int \mathcal{P}(ds' \mid s,a) \ V^{*}(s') \right\}
$$
so it is sufficient to pick a $\pi_{\epsilon}$ that solves the optimization problem up to $(1- \gamma) \ \epsilon$ accuracy of the supremum at each state $s$.
(if we find the maximizer, then $\epsilon = 0$)

For policy $\pi_{\epsilon}$, our earlier bound shows that
$$
||V^{*} - V^{\pi_{\epsilon}}||_{\infty}
\leq \epsilon
$$
This means that 
$$
V^{*}(s) \leq V^{\pi_{\epsilon}}(s) + \epsilon
\ , \quad \forall s \in \mathcal{S}
$$

Note that $V^{\pi_{\epsilon}}(s) \leq \sup_{\pi \in \Pi}(s)$ as $\pi_{\epsilon} \in \Pi$.
Taking $\epsilon \to 0$, we get
$$
\begin{align}
V^{*}(s)
&\leq \lim_{ \epsilon \to 0 } \{ V^{\pi_{\epsilon}}(s) + \epsilon \} \\[6pt]
&\leq \lim_{ \epsilon \to 0 } \{ \ \sup_{\pi \in \Pi}  
V^{\pi}(s) + \epsilon \ \} \\[6pt]
&= \sup_{\pi \in \Pi} V^{\pi }(s)
&\forall s \in \mathcal{S}
\end{align}
$$
Hence, we have proven that
$$
\boxed{ \ V^{*}(s) \leq \sup_{\pi \in \Pi} 
V^{\pi}(s) \ }
$$

---
**Direction-2**
Consider any $\pi \in \Pi$.
By the definition of $T^{\pi}$ and $T^{*}$, for any $V \in \mathcal{B}(\mathcal{S})$,
we have that for any $s \in \mathcal{S}$,
$$
\begin{align}
(T^{*}V)(s)
&= \int \pi(da \mid s) \ \left[ r(s,a) + \int 
\mathcal{P}(ds' \mid s,a) \ V(s') \right] \\[6pt]

&\leq \sup_{a \in \mathcal{A}}
\left\{  r(s,a) + \gamma \int  
\mathcal{P}(ds' \mid s,a) \ V(s') \right\} \\[6pt]

&= (T^{*}V)(s)
\end{align}
$$

With the choice of $V = V^{*}$, we have $T^{\pi}V^{*} \leq T^{*}V^{*}$.
As $T^{*}V^{*} = V^{*}$, we have
$$
T^{\pi}V^{*} \leq V^{*}
$$

Using the [[Monotonicity of Bellman Operator|monotonicity]] of $T^{\pi}$, we can conclude that
$$
T^{\pi}(T^{\pi} V^{*}) \leq T^{\pi}V^{*}
$$
By using our earlier bound from $V = V^{*}$, we can show that
$$
(T^{*})^{2} \ V^{*} \leq V^{*}
$$

We can repeat this argument $k$ times to get
$$
(T^{*})^{k} \ V^{*} \leq V^{*}
$$
As $k \to \infty$, the [[Uniqueness of Fixed Point]] shows that $(T^{\pi})^{k} \ V^{*}$ converges to $V^{\pi}$.
Note that the choice of $V^{*}$ is indeed irrelevant.

Hence we get
$$
V^{\pi} = \lim_{ k \to \infty } (T^{\pi})^{k} V^{*}
\leq V^{*}
$$

As this holds for any $\pi \in \Pi$, we can take the supremum over $\pi \in \Pi$ to get
$$
\boxed{ \ \sup_{\pi \in \Pi} V^{\pi} \leq V^{*} \ }
$$

---
**Conclusion**
Using the above $2$ directions, we can conclude that
$$
\boxed{ \ V^{*} = \sup_{\pi \in \Pi} V^{\pi} \ } \\
$$

---
## See Also
- [[Fixed Point]]
- [[Error Bounds of Bellman Equation]]
- [[Uniqueness of Fixed Point]]
- [[Monotonicity of Bellman Operator]]
- [[Bellman Operator]]
- [[Bellman Optimality Operator]]
