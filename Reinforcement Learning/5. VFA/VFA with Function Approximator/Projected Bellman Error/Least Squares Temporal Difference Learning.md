# Least Squares Temporal Difference Learning

---
### Problem Formulation
Let $\mathcal{F}$ be a linear [[Value Function Approximation|function approximation]] with basis functions(features): $\phi_{1}, \dots, \phi_{p}$.
$$
\mathcal{F}
= \{ s \mapsto \phi(s)^{T}w : w \in \mathbb{R}^{p} \}
$$

> **Goal**: Find a [[Value Function|value function]] that satisfies 

$$
V(s) = (\Pi_{\mathcal{F}, \mu} \ T^{\pi}V)(s)
\ , \quad \forall s \in \mathcal{S}
$$
> where $V$ is restricted to be in $\mathcal{F}$.

Instead of minimizing $||V - \Pi_{\mathcal{F}} \ T^{\pi}V||_{2, \mu}$ over [[Value Function|value functions]], we provide a direct solution similar to [[Solving Linear System of Equations|here]].

---
### Solving
For sake of simplicity, assume that $\mathcal{S}$ is finite and has $N$ states, potentially much larger than $p$.

Hence, each $\phi_{i}(i=1, \dots, p)$ is an N-dimensional vector.
Let $\Phi \in \mathbb{R}^{N \times p}$ be the matrix concatenating all features:
$$
\Phi = 
\begin{bmatrix}
\phi_{1} & \dots & \phi_{p}
\end{bmatrix}
$$
The [[Value Function|value function]] corresponding to a weight $w \in \mathbb{R}^{p}$ is defined as
$$
V_{N \times 1}  =\Phi_{N \times p} \ w_{p}
$$

Solving $V=(\Pi_{\mathcal{F}, \mu} \ T^{\pi}V)$ when $V=V(w)=\Phi \ w \in \mathcal{F}$ means that we have to find a $w \in \mathbb{R}$ such that
$$
\Phi w = \Pi_{\mathcal{F}, \mu} \ T^{\pi} \Phi w
$$
First, we re-write this in a matrix form.
Then, we solve it using linear algebraic manipulations.

---
### Vectorizing
**Vectorizing the Projection Operator**
To provide an explicit form for the projection operator, 
the $\mu$-weighted inner product between $V_{1}, V_{2} \in \mathbb{R}^{N}$ is
$$
\langle V_{1}, V_{2} \rangle_{\mu}
= \sum_{s \in \mathcal{S}} V_{1}(s) V_{2}(s) \ \mu(s)
= V_{1}^{T} \ MV_{2}
$$
where $M=\text{diag}(\mu)$.

The $L_{2}(\mu)$-[[Norm|norm]] $||V||_{2, \mu}$ can then be vectorized as 
$$
||V||^{2}_{2, \mu}
= \langle V, V \rangle_{\mu}
= \sum_{s \in \mathcal{S}} |V(s)|^{2} \ \mu(s)
= V^{T} M V
$$

The projection operator on linear $\mathcal{F}$ can then be denoted as 
$$
\Pi_{\mathcal{F}, \mu} V
= \arg \min_{V' \in \mathcal{F}}
||V' - V||^{2}_{2, \ \mu}
$$
This projected function $V = \Phi w^{+}$ where the weight $w^{+}$ minimizes
$$
\arg\min_{w \in \mathbb{R}^{p}} 
||\Phi w - V||^{2}_{2,\mu}
= \arg\min_{w \in \mathbb{R}^{p}}
(\Phi w - V)^{T} \ M \ (\Phi w - V)
$$

---
**Solving the Optimization**
Taking the derivative and setting it to zero, 
$$
\begin{align}
&\Phi^{T} M (\Phi w - V) = 0 \\[6pt]
\implies &w^{+} = (\Phi^{T} \ M \ \Phi)^{-1}
\ \Phi^{T} \ MV
\end{align}
$$
assuming that $\Phi^{T}M\Phi$ is invertible.

**Final Matrix Form**
Since the projected function is $\Phi w^{+}$, 
$$
\Pi_{\mathcal{F}, \mu} V
= \Phi ( \Phi^{T} M \Phi)^{-1}
\ \Phi^{T} \ MV
$$
We also have
$$
(T^{\pi} \ \Phi w)_{N \times 1}
= r^{\pi}_{N \times 1} + \gamma 
\mathcal{P}^{\pi}_{N \times N} \ \Phi_{N \times p}
\ w_{p}
$$

Combining all these, we get the matrix form of
$$
\Phi w
= [ \Phi (\Phi^{T} M \ \Phi)^{-1} \Phi^{T} M ]
[r^{\pi} + \gamma \mathcal{P}^{\pi} \ \Phi w ]
$$

There are two approaches on solving this: [[Solving Linear System of Equations|direct solution]] and [[Fixed Point Iteration (FPI)|fixed point iteration]].

---
## Direct Solution
Multiply both sides of 
$$
\Phi w
= [ \Phi (\Phi^{T} M \ \Phi)^{-1} \Phi^{T} M ]
[r^{\pi} + \gamma \mathcal{P}^{\pi} \ \Phi w ]
$$
by $\Phi^{T}M$ and simplifying, we get
$$
\begin{align}
\Phi^{T} M \ \Phi w
&= \Phi^{T} M \Phi \ (\Phi^{T} M \Phi)^{-1} 
\ \Phi^{T}M [r^{\pi} + \gamma \mathcal{P}^{\pi}  
\Phi w] \\[6pt]

\Phi^{T} M \ \Phi w
&= \Phi^{T}M \ [r^{\pi} + \gamma \mathcal{P}^{\pi} 
 \Phi w ] \\[6pt]
 
\Phi^{T}M \ [r^{\pi} + \gamma \mathcal{P}^{\pi} 
 \Phi w - \Phi w] &=  0
\end{align}
$$

Rearrange it to
$$
[\Phi^{T} M \Phi - \gamma \Phi^{T} M \mathcal{P}^{\pi} \Phi ] w
= \Phi^{T} M r^{\pi}
$$
Solving for $w$, we get
$$
w = [\Phi^{T} M (\Phi - \gamma \mathcal{P}^{\pi}
\Phi)]^{-1} \Phi^{T} M r^{\pi}
$$
This yields the population version of the [[Least Squares Temporal Difference Learning]] method.

---
### Geometric Intuition
![LSTD for PBE|300](https://notes-media.kthiha.com/Least-Squares-Temporal-Difference-Learning/7b5754429f157b6eae394db7f94f3337.png)

- $\Phi^{T}M \ [\gamma \mathcal{P}^{\pi} \ \Phi w - \Phi w] = 0$
- $\langle V_{1}, V_{2} \rangle_{\mu} = V^{T}_{1} \ M \  V_{2}$

$$
\langle \phi_{i}, \ T^{\pi} V(w) 
- V(w) \rangle_{\mu}
= \langle \phi_{i}, \text{BR}(V(w)) \rangle_{\mu}
= 0 \ , \quad \forall i = 1, \dots, p
$$

[[Least Squares Temporal Difference Learning|LSTD]] finds a $w$ such that the [[Bellman Residual Minimization (BRM)|Bellman Residual]] is [[Orthographic Projection|orthogonal]] to the basis of $\mathcal{F}$.

---
## Error Bound
Suppose that we find
$$
V = \Pi_{\mathcal{F}, \ \rho^{\pi}} \ T^{\pi}V
$$
For a [[Linear Function Approximation|linear function approximation]], the [[Least Squares Temporal Difference Learning|LSTD method]] and the [[Fixed Point Iteration for Projected Bellman Operator|fixed point iteration]] find this solution.
Let's denote this the [[Temporal Difference Learning for Policy Evaluation(TD)|TD solution]] $V_{TD}$.

How close is this [[Value Function|value function]] to the true [[Value Function|value function]] $V^{\pi}$?
- If the [[Value Function|value function space]] $\mathcal{F}$ cannot represent $V^{\pi}$ precisely (which is often the case under [[Value Function Approximation|function approximation]]), we cannot expect to have a small error
- The smallest error is $||\Pi_{\mathcal{F}, \rho^{\pi}} V^{\pi} - V^{\pi}||$.
- The [[Temporal Difference Learning for Policy Evaluation(TD)|TD solution]] is not as close to $V^{\pi}$ as the projection of $V^{\pi}$ onto $\mathcal{F}$, but it can be close to that.

---
> **Proposition**
> If $\rho^{\pi}$ is the stationary distribution of $\pi$, we have

$$
||V_{TD} - V^{\pi}||_{2, \ \rho^{\pi}}
\leq \frac{||\Pi_{\mathcal{F}, \ \rho^{\pi}} 
\ V^{\pi} - V^{\pi} ||_{2, \rho^{\pi}}}
{\sqrt{ 1 - \gamma^{2} }}
$$

---
