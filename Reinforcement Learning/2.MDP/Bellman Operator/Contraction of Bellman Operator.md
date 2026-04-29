# Contraction of Bellman Operator
#rl/bellman-equation/operator/contraction

> **Lemma**
For any $\pi$, the [[Bellman Operator]] $T^{\pi}$ is a $\gamma$-[[Contraction Mapping|contraction mapping]].
The [[Bellman Optimality Operator]] $T^{*}$ is a $\gamma$-[[Contraction Mapping|contraction mapping]].

For any $V_{1}, \ V_{2} \in \mathcal{B}(\mathcal{S})$ and $T$ being either $T^{\pi}$ or $T^{*}$,
$$
|| \ T V_{1} - TV_{2} \ ||_{\infty}
\leq \gamma \ ||V_{1} - V_{2} ||_{\infty}
$$

For any $Q_{1}, \ Q_{2} \in \mathcal{B}(\mathcal{S} \times \mathcal{A})$ and $T$ being either $T^{\pi}$ or $T^{*}$,
$$
|| \ T Q_{1} - TQ_{2} \ ||_{\infty}
\leq \gamma \ ||Q_{1} - Q_{2} ||_{\infty}
$$

---
## Proof
We shall show it for the Bellman operator $T^{\pi}: \mathcal{B}(\mathcal{S} \times \mathcal{A}) \to \mathcal{B}(\mathcal{S} \times \mathcal{A})$.

Consider two arbitrary [[Quality Function|action-value functions]] $Q_{1}, Q_{2} \in \mathcal{B}(\mathcal{S} \times \mathcal{A})$.
Let the [[Metric|metric]] be $d(Q_{1}, Q_{2}) = ||Q_{1} - Q_{2}||_{\infty}$.

For any $(s,a) \in \mathcal{S} \times \mathcal{A}$, we have
$$
\begin{align}
&|(T^{\pi} Q_{1})(s,a) - (T^{\pi}Q_{2})(s,a)| \\[6pt]

&= |\left[  r(s,a) + \gamma \int  
\mathcal{P}(ds' \mid s,a) \ \pi(da' \mid s') 
\ Q_{1}(s', a') \right]| \\[6pt]

&- |\left[  r(s,a) + \gamma \int  
\mathcal{P}(ds' \mid s,a) \ \pi(da' \mid s') 
\ Q_{2}(s', a') \right]|  \\[6pt]

&= \gamma \ \left| \int \mathcal{P}(ds' \mid s,a)  
\ \pi(da' \mid s') \ (Q_{1}(s', a') - Q_{2}(s', a')) 
\right|
\end{align}
$$

**Getting the upper bound**
Let us upper bound the right hand side.
We have an integral of the form $\left| \int P(dx) f(x) \right|$ (or summation $\left| \sum_{x} P(x)\ f(x) \right|$ for a countable state space).

This can be upper bounded as
$$
\begin{align}
\left| \int P(dx) \ f(x) \right|
&\leq \int |\ P(x) \ f(x) \ | \\[6pt]
&= \int |P(x)| \cdot |f(x)| \\[6pt]
&\leq \int P(dx) \cdot \sup_{x \in \mathcal{X}}
|f(x)| \\[6pt]
&= \sup_{x \in \mathcal{X}} |f(x)|
\ \int P(dx) \\[6pt]
&= ||f||_{\infty}
\end{align}
$$
where in the last equality, we used $\int P(dx) = 1$.

**Applying the upper bound**
Using the upper bound above, we get
$$
\begin{align}
&|\  (T^{\pi} Q_{1})(s,a) - (T^{\pi} Q_{2})(s,a) \ |
\\[6pt]

&= \gamma \left| \ \int \mathcal{P}(ds' \mid s,a) 
\ \pi(da' \mid s') \ (Q_{1}(s',a') - Q_{2}(s', a'))  
\ \right| \\[6pt]

&\leq \gamma \int \mathcal{P}(ds' \mid s,a)
\ \pi(da' \mid s') \ 
| Q_{1}(s', a') - Q_{2}(s', a') | \\[6pt]

&\leq \gamma ||Q_{1} - Q_{2}||_{\infty}
\int \mathcal{P}(ds' \mid s,a) \pi(da' \mid s')  
\\[6pt]

&= \gamma ||Q_{1} - Q_{2}||_{\infty}
\end{align}
$$

**Applying the inequality to supremum**
This inequality holds for **any** $(s,a) \in \mathcal{S} \times \mathcal{A}$, so it holds for its supremum over $\mathcal{S} \times \mathcal{A}$ too.
$$
|| \ (T^{\pi}Q_{1}) - (T^{\pi}Q_{2}) \ ||_{\infty}
\leq \gamma \ || \ Q_{1} - Q_{2} \||_{\infty}
$$
Hence, this proves that $T^{\pi}$ is a [[Contraction Mapping|contraction]].

---
## Proof for Bellman optimality Operator
The proof for the [[Bellman Optimality Operator]] is similar.

First given two functions $f_{1}, f_{2}: \mathcal{A} \to \mathbb{R}$ let $(1)$ be
$$
\left| \max_{a \in \mathcal{A}} f_{1}(a)
- \max_{a \in \mathcal{A}} f_{2}(a) \right|
\leq \max_{a \in \mathcal{A}} 
\left| f_{1}(a)- f_{2}(a) \right|
$$

Now consider two [[Quality Function|action-value functions]] $Q_{1}, Q_{2} \in \mathcal{B}(\mathcal{S} \times \mathcal{A})$.
$$
\begin{align}
&| \ (T^{*}Q_{1})(s,a) - (T^{*}Q_{2})(s,a)\ |  
\\[6pt]

&= \gamma \ \left| \int \mathcal{P}(ds' \mid s,a) 
\ \left( \max_{a' \in \mathcal{A}} Q_{1}(s', a') 
- \max_{a' \in \mathcal{A}} Q_{2}(s', a')  
\right) \right| \\[6pt]

&\leq \gamma \ \int \mathcal{P}(ds' \mid s,a) 
\ \sup_{s' \in \mathcal{S}} \left| \max_{a' \in \mathcal{A}} Q_{1}(s', a') 
- \max_{a' \in \mathcal{A}} Q_{2}(s', a')  
\right|  \\[6pt]

&\leq \gamma \ \int \mathcal{P}(ds' \mid s,a) 
\ \sup_{s' \in \mathcal{S}}  
\max_{a' \in \mathcal{A}} \left|  Q_{1}(s', a') 
-  Q_{2}(s', a')  \right|  \quad \text{by (1)}  
\\[6pt]

&= \gamma \ \sup_{(s,a) \in  
\mathcal{S} \times \mathcal{A}}
| \ Q_{1}(s,a) - Q_{2}(s,a) \ |  
\int \mathcal{P}(ds' \mid s,a) \\[6pt]

&= \gamma \ ||Q_{1} - Q_{2}||_{\infty}
\end{align}
$$

This inequality holds for **any** $(s,a) \in \mathcal{S} \times \mathcal{A}$, so it holds for its supremum over $\mathcal{S} \times \mathcal{A}$ too.
$$
|| \ (T^{*}Q_{1}) - (T^{*}Q_{2}) \ ||_{\infty}
\leq \gamma \ || \ Q_{1} - Q_{2} \||_{\infty}
$$
Hence, this proves that $T^{*}$ is a [[Contraction Mapping|contraction]].

---
> **Note**
> Note that [[Bellman Operator]] are $\gamma$-[[Contraction Mapping|contraction]] $w.r.t$ the [[Infinity Norm|supremum norm]]. This may not hold for other [[p-Norm|norms]].
> 
> It is not generally true that for all [[Markov Decision Process (MDP)|MDPs]] and any choice of distribution $v$ that $|| \ TV_{1} - TV_{2} \ ||_{2,v} \leq \gamma ||\ V_{1} - V_{2}\ ||_{2,v}$.

---
## See Also
- [[Contraction Mapping]]
- [[Bellman Equation for Quality Function]]
- [[Bellman Equation for Optimal Quality Functions]]