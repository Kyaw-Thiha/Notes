# Stationary Distribution of Policy
> The stationary distribution of a [[Policy|policy]] $\pi$ is the distribution that does not change as we follow $\pi$.

## Formal Definition
Assume that we initiate the agent at $S_{1} \sim \rho \in \mathcal{M}(\mathcal{S})$.
The agent follows $\pi$, and gets to $S_{2} \sim \mathcal{P}^{\pi}(\cdot \mid S_{1})$.

The probability distribution of $S_{2}$ being in a set $B$ is
$$
\mathbb{P}\{ S_{2} \in B \}
= \int p(ds) \ \mathcal{P}^{\pi}(B \mid s)
$$
or for countable state space, the probability of being in state $s'$ is
$$
\mathbb{P}\{ S_{2} = y \}
= \sum_{s \in \mathcal{S}}
p(s) \mathcal{P}^{\pi}(s' \mid s)
$$

If the distribution of $S_{1}$ and $S_{2}$ are both $p^{\pi}$, we say that $p^{\pi}$ is the [[Stationary Distribution of Policy|stationary distribution]] induced by $\pi$.

---
By induction, it would be distribution of $S_{3}, \ S_{4}, \ \dots$ too.
If $S_{1}$ and $S_{2}$ are both at the stationary distribution, we have
$$
\mathbb{P}\{ S_{1} = s' \}
= \rho^{\pi}(s')
= \sum_{s \in \mathcal{S}} \mathcal{P}^{\pi}
(s'\mid s) \ \rho^{\pi}(s)
= \mathbb{P}\{ S_{2}=s' \}
$$
or 
$$
\rho^{\pi}(B)
= \int \rho^{\pi}(ds) \ \mathcal{P}^{\pi}(B \mid s)
$$

---
### Vectorizing
For countable state spaces, we can write it in the matrix form.
If we denote $\mathcal{P}^{\pi}$ by an $n \times n$ matrix with $[\mathcal{P}^{\pi}]_{s \ s'} = \mathcal{P}^{\pi}(s' \mid s)$, we have
$$
\rho^{\pi}(s') = \sum_{s} \mathcal{P}^{\pi}_{s \ s'} 
\rho^{\pi}_{s} \quad \ \forall s' \in \mathcal{S}
$$
So, 
$$
\rho^{\pi \ T} = \rho^{\pi \ T} \mathcal{P}^{\pi}
$$
The distribution $\rho^{\pi}$ is the left eigenvector corresponding to the eigenvalue with value $1$ of matrix $\mathcal{P}^{\pi}$.

---
## Convergence to Stationary Policy
[[Markov Models|Markov chain]] induced by $\pi$ converges to the stationary distribution $\rho^{\pi}$, under certain conditions even if the initial distribution is not $\rho^{\pi}$.

For any $\mu \in \mathcal{M}(\mathcal{S})$, we have that 
$$
\mu(\mathcal{P}^{\pi})^{k} \to \rho^{\pi}
$$

---
## Contraction Property
**Lemma**: The [[Bellman Operator]] $T^{\pi}$ is a $\gamma-$[[Contraction Mapping|contraction]] $w.r.t$ $||\cdot||_{2, \rho^{\pi}}$.
**Proof**: 
$$
\begin{align}
&||T^{\pi} V_{1} - T^{\pi}V_{2}||^{2}_{2, \ 
\rho^{\pi}} \\[6pt]

&= \int \rho^{\pi}(ds) \left| \left(  r^{\pi}(s)  
+ \gamma \int \mathcal{P}^{\pi}(ds' \mid s) V_{1}(s') \right) 
- \left( r^{\pi}(s) + \gamma \int \mathcal{P}^{\pi} (ds' \mid s) V_{2}(s') \right) \right|^{2} \\[6pt]

&= \int \rho^{\pi}(ds) \left| \gamma \int  
\mathcal{P}^{\pi}(ds' \mid s) \ (V_{1}(s')  
- V_{2}(s')) \right|^{2} \\[6pt]

&\leq \gamma^{2} \int \rho^{\pi}(ds)  
\ \mathcal{P}^{\pi}(ds' \mid s)
\ |V_{1}(s') - V_{2}(s')|^{2}  
\ \quad \text{by (1)} \\[6pt]

&= \gamma^{2} \int \rho^{\pi}(ds') 
|V_{1}(s') - V_{2}(s')|^{2} 
\ \quad \text{by (2)} \\[6pt]

&= \gamma^{2} ||V_{1} - V_{2}||^{2}_{2, \rho^{\pi}}
\end{align}
$$
where
- $(1)$: by [[Jensen Inequality]]
- $(2)$: by definition of the [[Stationary Distribution of Policy|stationary distribution]]

---
