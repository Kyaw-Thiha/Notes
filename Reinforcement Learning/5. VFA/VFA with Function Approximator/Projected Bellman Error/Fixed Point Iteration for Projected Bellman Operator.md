# Fixed Point Iteration for Projected Bellman Operator

## Background
Recall from [[Projected Bellman Error (PBE)|here]] that we designed two iterative approaches for finding the [[Fixed Point|fixed point]] of $\Pi_{\mathcal{F}, \mu}T^{\pi}$.

We designed methods similar to [[Stochastic Approximation(SA)|stochastic approximation]].
Since we not using the true model, we can handle the noise.

We specifically focus on the case when the distribution $\mu$ is the [[Stationary Distribution of Policy|stationary distribution]] $\rho^{\pi}$ of $\pi$.

---
## Approach-1
Consider the [[Fixed Point Iteration (FPI)|update rule]]
$$
\hat{V}_{k+1} \leftarrow (1 - \alpha) \hat{V}_{k}
+ \alpha \Pi_{\mathcal{F}, \rho^{\pi}} 
\ T^{\pi} \hat{V}_{k}
$$
with an $0 < \alpha \leq 1$

This can be seen as a [[Fixed Point Iteration (FPI)|fixed-point iterative method]] with the operator

$$
L: V \mapsto [(1 - \alpha) \ \mathbf{I} 
+ \alpha \ \Pi_{\mathcal{F}, \rho^{\pi}} T^{\pi}] \ V
$$
**Convergence**:
This operator is a [[Contraction Mapping|contraction]] $w.r.t$ $L_{2}(\rho^{\pi})$:
$$
||LV_{1} - LV_{2}||_{2, \ \rho^{\pi}}
\leq (1 - \alpha) ||V_{1} - V_{2}||_{2, \ \rho^{\pi}}
+ \alpha ||\Pi_{\mathcal{F}, \rho^{\pi}} 
\ T^{\pi}(V_{1} - V_{2}) ||_{2, \rho^{\pi}}
$$

- The projection operator $\Pi_{\mathcal{F}, \ \rho^{\pi}}$ is non-expansive $w.r.t$ the $L_{2}(\rho^{\pi})$
- This allows [[Norm|norm]] in second term of RHS to be upper-bounded.
- The [[Bellman Operator|Bellman operator]] $T^{\pi}$ is a $\gamma$-[[Contraction Mapping|contraction]] $w.r.t$ the same [[Norm|norm]]:

$$
\begin{align}
&||\Pi_{\mathcal{F}, \ \rho^{\pi}} 
\ T^{\pi}(V_{1} - V_{2})||_{2, \ \rho^{\pi}} \\[6pt]

\leq \ &|| T^{\pi}(V_{1} - V_{2}) ||_{2, \rho^{\pi}} \\[6pt]

\leq \ & \gamma \ || V_{1} - V_{2} ||_{2, \rho^{\pi}} \\[6pt]
\end{align}
$$

This along with the prior contraction property shows that
$$
||LV_{1} - LV_{2}||_{2, \ \rho^{\pi}}
\leq [(1 - \alpha) + \alpha \gamma] 
\ ||V_{1} - V_{2}||_{2, \ \rho^{\pi}}
$$
Hence if $0 \leq \alpha \leq 1$, $L$ is a [[Contraction Mapping|contraction]].

Therefore, the iterative method 
$$
\hat{V}_{k+1} \leftarrow (1 - \alpha) \hat{V}_{k}
+ \alpha \Pi_{\mathcal{F}, \rho^{\pi}} 
\ T^{\pi} \hat{V}_{k}
$$
is going to be convergent.

> Note that its projection operator is $w.r.t$ $||\cdot||_{2, \ \rho^{\pi}}$.
> The convergence property may not hold for other $\mu \neq \rho^{\pi}$.

---
**Setting things up**
Let's use a linear [[Value Function Approximation|function approximation]]:
$$
\hat{V}_{k} = \Phi w_{k}
$$
We use the explicit projection function as defined in [[Least Squares Temporal Difference Learning]]:
$$
\Pi_{\mathcal{F}, \ \mu} V
= \Phi(\Phi^{T} M \ \Phi)^{-1}
\ \Phi^{T} \ MV
$$

We will be using $D = \text{diag}(\rho^{\pi})$ instead of $M$ to emphasize the dependence on $\rho^{\pi}$.

**Deriving**
The iteration $\hat{V}_{k+1} \leftarrow (1 - \alpha) \hat{V}_{k} + \alpha \Pi_{\mathcal{F}, \rho^{\pi}} \ T^{\pi} \hat{V}_{k}$ can be rewritten as
$$
\begin{align}
\hat{V}_{k+1} = \Phi \ w_{k+1}
\leftarrow \ &(1 - \alpha) \ \Phi w_{k} \\
&+ \alpha \Phi \ (\Phi^{T} D^{\pi} \Phi)^{-1}
\ \Phi^{T} D^{\pi} \ 
[r^{\pi} + \gamma \mathcal{P}^{\pi} \Phi w_{k}]
\end{align}
$$

Multiplying both sides by $\Phi^{T}D^{\pi}$,
$$
\begin{align}
(\Phi^{T} D^{\pi} \Phi) \ w_{k+1}
\leftarrow \ &(1 - \alpha)(\Phi^{T} D^{\pi} \Phi) \ w_{k} \\[6pt]
&+ \alpha (\Phi^{T} D^{\pi} \Phi) 
(\Phi^{T} D^{\pi} \Phi)^{-1} \ \Phi^{T} D^{\pi} \ 
[r^{\pi} + \gamma \mathcal{P}^{\pi} \ \Phi w_{k}]
\end{align}
$$

Assuming that $\Phi^{T} D^{\pi} \Phi$ is invertible,
$$
w_{k+1} \leftarrow (1 - \alpha) \ w_{k}
+ \alpha (\Phi^{T} D^{\pi} \Phi)^{-1}
\ \Phi^{T} D^{\pi} \ 
[r^{\pi} + \gamma \mathcal{P}^{\pi} \ \Phi w_{k}]
$$
This is a convergent iteration, and converges to a [[Fixed Point|fixed point]] of
$$
\Phi w = \Pi_{\mathcal{F}, \ \mu} 
\ T^{\pi} \Phi w
$$

**Discussion**
$$
w_{k+1} \leftarrow (1 - \alpha)w_{k}
+ \alpha (\Phi^{T} D^{\pi} \Phi)^{-1}
\ \Phi^{T} D^{\pi} \ 
[r^{\pi} + \gamma \mathcal{P}^{\pi} \ \Phi w_{k}]
$$
- This requires a one-time inversion of a $p \times p$ matrix
$$
(\Phi^{T} D^{\pi} \Phi)
= \sum_{s} \rho^{T}(s) \ \phi^{T}(s) \ \phi(s)
$$
	which is $O(p^{3})$ operation.
- A matrix-vector multiplication at every time step $O(p^{2})$.
- In online setting, this matrix is updated as new data arrives.
  A naive approach of updating the matrix and re-computing its inverse would be costly.

---
## Solution-2
Recall that from [[Least Squares Temporal Difference Learning]], 
$$
\Phi^{T} D^{\pi} \ [r^{\pi} 
+ \gamma \mathcal{P}^{\pi} \Phi w - \Phi w] = 0
$$
where  $D = \text{diag}(\rho^{\pi})$ instead of $M$ to emphasize dependence on $\rho^{\pi}$.

Hence, this yields same solution as [[Least Squares Temporal Difference Learning|LSTD]].
- If $Lw = 0$, we also have $\alpha Lw = 0$.
- Adding an identity to both sides does not change the equation.
$$
w + \alpha \ Lw = w
$$
- This is in the form of a [[Fixed Point Theorem (FPT)|fixed-point equation]] for a new operator:
$$
L': w \mapsto (\mathbf{I} - \alpha L) \ w
$$
- The fixed point of $L'$ is the same as the solution of $Lw=0$.
- We may apply $w_{k+1} \leftarrow L' w_{k} = (\mathbf{I} + \alpha L) w_{k}$, 
  assuming $L'$ is a [[Contraction Mapping|contraction]].

If we choose
$$
L:w \mapsto \phi^{T} D^{\pi} [r^{\pi} 
+ \gamma \mathcal{P}^{\pi} \ \Phi w]
$$
we get the following iterative procedure:
$$
w_{k+1}
\leftarrow w_{k} + \alpha \Phi^{T} D^{\pi}
[r^{\pi} + \gamma \mathcal{P}^{\pi} \ \Phi w_{k}
- \Phi w_{k}]
$$
---

