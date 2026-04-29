# Bellman Error Bound wrt Stationary Distribution

> **Proposition**:
> Let $\rho^{\pi}$ be the [[Stationary Distribution of Policy|stationary distribution]] of $\mathcal{P}^{\pi}$.
> For any $V \in \mathcal{B(S)}$ and $p\geq 1$, we have [[Bellman Error|Bellman error bound]] of

$$
||V - V^{\pi}||_{1, \rho^{\pi}}
\leq \frac{||V - T^{\pi}V||_{p, \ p^{\pi}}}
{1 - \gamma}
$$

**Remark**: This is similar to [[Bellman Error]] $||V-V^{\pi}||_{\infty} \leq \frac{||V - T^{\pi}V||_{\infty}}{1 - \gamma}$, but $w.r.t$ the [[Stationary Distribution of Policy|stationary distribution]] $\rho^{\pi}$.

---
## Proof
For any $V$, we have that 
$$
\begin{align}
V - V^{\pi}
&= V - T^{\pi}V + T^{\pi}V - V^{\pi} \\[6pt]
&= (V - T^{\pi}V) + (T^{\pi}V - T^{\pi}V^{\pi})
\end{align}
$$
The second term on the RHS evaluated at state $s$ is
$$
(T^{\pi}V)(s) - (T^{\pi}V^{\pi})(s)
= \gamma \int \mathcal{P}^{\pi}(ds' \mid s)
(V(s') - V^{\pi}(s'))
$$
Taking the absolute value and integrating $w.r.t$ $\rho^{\pi}$, 
$$
\begin{align}
\int |V(s) - V^{\pi}| \ d\rho^{\pi}(s)
&\leq \int |V(s) - (T^{\pi}V)(s)| \ d\rho^{\pi}(s)
\\[6pt]
&+ \gamma \int d\rho^{\pi}(s)
\left| \int \mathcal{P}^{\pi}(ds' \mid s) 
(V(s') - V^{\pi}(s'))  \right|
\end{align}
$$
By [[Jensen Inequality]], we have
$$
\begin{align}
\int |V(s) - V^{\pi}(s)| \ d\rho^{\pi}(s)
&\leq \int |V(s) - (T^{\pi}V)(s)| \ d\rho^{\pi}(s)
\\[6pt]
&+ \gamma \int d\rho^{\pi}(s) \ \mathcal{P}^{\pi} 
(ds' \mid s) \ |V(s') - V^{\pi}(s')|
\end{align}
$$

Because $\rho^{\pi}$ is the [[Stationary Distribution of Policy|stationary distribution]], the second integral in the RHS can be simplified as 
$$
\int d\rho^{\pi}(s) \ \mathcal{P}^{\pi}(ds' \mid s)
\ |V(s) - V^{\pi}(s')|
= \int d\rho^{\pi}(s) \ |V(s') - V^{\pi}(s')|
$$
So, 
$$
||V - V^{\pi}||_{1, \rho^{\pi}}
\leq ||V - T^{\pi}V||_{1, \rho^{\pi}}
+ \gamma ||V - V^{\pi}||_{1, \rho^{\pi}}
$$
After re-arranging, we get the result for $p=1$.
By [[Jensen Inequality]], we have that 
$$
||V - T^{\pi}V||_{1, \rho^{\pi}}
\leq ||V - T^{\pi}V||_{p, \rho^{\pi}}
$$
for any $p \geq 1$.

---
