# Temporal Difference Learning
## Background
[[Monte Carlo Estimation for Policy Evaluation|Monte Carlo methods]] estimate $V^{\pi}(s)$ by using returns $G^{\pi}(s)$.
But it does not benefit from recursive property of the [[Value Function|value function]] since it is agnostic to the [[Markov Decision Process (MDP)|MDP structure]].

Recall that in [[Planning (Reinforcement Learning)]], we discussed methods benefiting from the structure of [[Markov Decision Process (MDP)|MDP]].
Now we will consider using similar methods but without knowing $\mathcal{P}$ and $\mathcal{R}$.

---
## Derivation
Recall the [[Value Iteration|Value Iteration algorithm]] for policy evaluation:
At state $s$, the procedure is
$$
V_{k+1} \leftarrow
r^{\pi}(s) + \gamma \int \mathcal{P}(ds' \mid s,a)
\ \pi(da \mid s) \ V_{k}(s')
$$
If we do not know $r^{\pi}$ and $\mathcal{P}$, we cannot compute this.

Suppose that we have $n$ samples of actions $A_{i} \sim \pi(\cdot \mid s)$, states $S_{i}' \sim \mathcal{P}(\cdot \mid s, \ A_{i})$, and rewards $\mathcal{R_{i}} \sim \mathcal{R}(\cdot \mid s,A_{i})$.
Using these samples and $V_{k}$, we compute
$$
Y_{i} = R_{i} + \gamma V_{k}(S_{i}')
$$
Now note that 
$$
\mathbb{E}[R_{i} \mid S=s] = r^{\pi}(s)
$$
and
$$
\mathbb{E}[V_{k}(S'_{i}) \mid S=s]
= \int \mathcal{P}(ds' \mid s,a) \ \pi(da \mid s)
\ V_{k}(s')
$$
so the $r.v.$ $Y_{i}$ satisfies
$$
\mathbb{E}[Y_{i} \mid S=s]
= \mathbb{E}[R_{i} + \gamma V_{k}(S'_{i}) \mid S=s]
= (T^{\pi}V_{k})(s)
$$
This means $Y_{i}$ is an unbiased sample from the effect of [[Bellman Operator|bellman operator]] $T^{\pi}$ on [[Value Function|value function]] $V_{k}$ evaluated at $s$.

Hence, 
- we can use the [[Sample Average Estimator|sample mean]] to estimate $(T^{\pi}V_{k})(s)$
- or devise a [[Stochastic Approximation(SA)|SA procedure]].

---
## Synchronous Temporal Difference Learning
If we were to solve using [[Sample Average Estimator]], 
$$
V_{k+1}(s) \triangleq \frac{1}{n} 
\sum^{n}_{i=1} Y_{i}
$$
converges to $(T^{\pi}V_{k})(s)$ by [[Weak Law of Large Numbers (LLN)|LLN]].

Instead, we could also use the [[Stochastic Approximation(SA)|SA Procedure]] and update as
$$
V_{k+1, \ j+1}(s)
= (1 - \alpha_{j}(s)) \ V_{k+1,\ j}(s)
+ \alpha_{j}(s) Y_{j}
\quad  \ j=1,2,\dots
$$
This is `Synchronous Temporal Difference Learning`:
> Require: Policy $\pi$, step size schedule $(\alpha_{k})_{k \ \geq \ 1}$.
> Initialize $V_{1}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ arbitrary
> for iteration $k=1,2,\dots$ do
> - for each state $s \in \mathcal{S}$ do
> 	- Let $A \sim \pi(\cdot \mid s)$
> 	- $S'(s) \sim \mathcal{P}(\cdot \mid S,A)$ and $R(s) \sim \mathcal{R}(\cdot \mid s,A)$
> 	- Let $(\hat{T}^{\pi}V_{k})(s) \triangleq R(s) + \gamma V_{k}(S'(s))$
> - end for
> - Update
$$
V_{k+1} \leftarrow (1 - \alpha_{k}) \ V_{k} 
+ \alpha_{k} \ \hat{T}^{\pi} V_{k}
$$

---
### Empirical Bellman Operator
The empirical [[Bellman Operator]] $\hat{T}^{\pi}$ is defined as
$$
(\hat{T}^{\pi}V_{k})(s) \triangleq R(s) + \gamma V_{k}(S'(s))
$$
It provides an unbiased estimate of $(T^{\pi}V_{k})(s)$:
$$
\mathbb{E}[(\hat{T}V_{k})(s) \mid S=s]
= (T^{\pi}V_{k})(s)
$$

### Empirical Value Iteration
The empirical version of the [[Value Iteration|value iteration algorithms]] is
$$
V_{k+1} \leftarrow \hat{T}^{\pi} V_{k}
= T^{\pi}V_{k} + \underbrace{ \left( 
\hat{T}^{\pi}V_{k} - T^{\pi}V_{k}
\right)}_{\triangleq \epsilon_{k}}
$$
This update can be decomposed to 
- `Deterministic Part`: $T^{\pi}V_{k}$ (usual [[Value Iteration|VI]])
- `Stochastic Part`: $\hat{T}^{\pi}V_{k} - T^{\pi}V_{k}$ (zero-mean $r.v.$)

---
## Asynchronous Temporal Difference Learning
Now, let's consider not updating all states at same time.

Then, we get `Asynchronous Temporal Difference Learning`:
> **Require**: Policy $\pi$, step size schedule $(\alpha_{k})_{k \ \geq \ 1}$.
> Initialize $V_{1}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ arbitrary 
> Initialize $S_{1} \sim \rho$
> - **for** each step $t=1,2,\dots$ **do**
> 	- Let $A_{t} \sim \pi(\cdot \mid s)$
> 	- Take action $A_{t}$, observe $S_{t+1} \sim \mathcal{P}(\cdot \mid S_{t}, \ A_{t})$ and $R_{t} \sim \mathcal{R}(\cdot \mid S_{t}, A_{t})$
> 	- Update
$$
V_{t+1}(s) \leftarrow \begin{cases}
V_{t}(s) + \alpha_{t}(s) [ \ R_{t}  
+ \gamma \ V_{t}(S_{t+1}) - V_{t}(S_{t}) \ ]
& s=S_{t} \\[6pt]
V_{t}(s) & s \neq S_{t}
\end{cases}
$$
> - **end for**

The update rule could be written in simpler but less precise form:
$$
V(S_{t}) \leftarrow V(S_{t}) + \alpha_{t}(S_{t})
[R_{t} + \gamma V(S_{t+1}) - V(S_{t})]
$$
without showing any explicit dependence of $V$ on time $t$.

---
## See Also
- [[Temporal Difference(TD) Error]]
- [[TD Learning for Action-Value Function]]
- [[Value Iteration]]
- [[Stochastic Approximation(SA)]]
- [[Sample Average Estimator]]
- [[Bellman Operator]]