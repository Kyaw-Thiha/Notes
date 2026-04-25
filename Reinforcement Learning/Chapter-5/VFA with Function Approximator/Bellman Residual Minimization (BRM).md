# Bellman Residual Minimization
#rl/vfa/bellman-residual-minimization

[[Bellman Residual Minimization (BRM)|BRM]] directly go for the solution of the [[Fixed Point|fixed-point equation]].
$$
V \leftarrow \arg\min_{V \in \mathcal{F}}
||V - T^{\pi}V||_{p,\mu}
$$

![BRM|400](https://notes-media.kthiha.com/Bellman-Residual-Minimization-(BRM)/b9e6ef695258aad108f655a65b680add.png)

---
## Motivation
Recall that [[Approximate Value Iteration (AVI)|AVI]] tries to benefit from the [[Contraction of Bellman Operator|contraction property]] of [[Bellman Operator|Bellman operator]], which allowed [[Value Iteration|VI]] to be a sound solution.

However, this convergence may not always work under [[Value Function Approximation|FA]].
We aim to benefit from another structural property of [[Markov Decision Process (MDP)|MDP]].

---
## Background
We know from [[Bellman Operator|prior theorem]] that if we find a $V$ such that $V=T^{\pi}V$, that function must equal $V^{\pi}$.

Under [[Value Function Approximation]], we may not achieve this exact equality but instead have
$$
V \approx T^{\pi}V
$$
for some $V \in \mathcal{F}$.

So, we need to quantify the quality of this approximation.
The $L_{p}$[[p-Norm|-norm]] $w.r.t$ a distribution $\mu$ is a common choice:
$$
V \leftarrow \arg\min_{V \in \mathcal{F}}
||V - T^{\pi}V||_{p,\mu}
= || \ \text{BR}(V) \ ||_{p,\mu}
$$
where 
- the value of $p$ is often selected as $2$.
- $\mu$ might be the distribution induced by a behaviour policy.

This process is called the [[Bellman Residual Minimization (BRM)]].
The same procedure works for the [[Quality Function|action-value function]] $Q$ with obvious changes.

---
> **Comparism to [[Approximate Value Iteration (AVI)]]**
> 
> Compared to [[Approximate Value Iteration (AVI)|AVI]], we do not mimic the iterative process of [[Value Iteration|VI]].
> Instead, [[Bellman Residual Minimization (BRM)|BRM]] directly go for the solution of the [[Fixed Point|fixed-point equation]].

---
## Geometric Version
When $\mathcal{F}$ is the set of linear functions, its geometry is the subspace spanned by $\phi$.
![image|400](https://notes-media.kthiha.com/Bellman-Residual-Minimization-(BRM)/b9e6ef695258aad108f655a65b680add.png)

Given $V \in \mathcal{F}$, we apply $T^{\pi}$ to it to get $T^{\pi}V$.
In general $T^{\pi}V$ is not within $\mathcal{F}$, so we visualize it with a point outside the plane.

[[Bellman Residual Minimization (BRM)|BRM]] minimizes distance $||V - T^{\pi}V||_{2, \mu}$ among all functions in $V \in \mathcal{F}$.

---
**Zero Error Case**:
If there exists $V \in \mathcal{F}$ that makes $||V - T^{\pi}V||_{2,\mu} = 0$ 
and if we assume that $\mu(x)>0$ for all $x \in \mathcal{X}$, we can conclude
$$
V(x) = (T^{\pi}V)(x)
\ \quad \ \text{for } x \in \mathcal{X}
$$
This is the [[Bellman Equation for Value Function|Bellman equation]], so its solution is $V=V^{\pi}$.

---
**Non-Zero Error Case**:
In general, the error $||V - T^{\pi}V||_{2,\mu} \neq 0$.
so, the minimizer $V \leftarrow \arg\min_{V \in \mathcal{F}} ||V - T^{\pi}V||_{p,\mu}$ does not  result in [[Value Function|value function]] $V^{\pi}$.
Nonetheless, it still have some good approximation properties.

---
## Bellman Error-based Error Bound
Recall that the [[Bellman Error]] is defined as
$$
||V - V^{\pi}||_{\infty}
\leq \frac{||V - T^{\pi}V||_{\infty}}{1 - \gamma}
$$
This [[Bellman Error]] can be used as a surrogate loss.
- Instead of minimizing $||V - V^{\pi}||$, we minimize its upper bound $||V - T^{\pi}V||$.
- If the upper bound is small, the quantity of interest is small.

Note that this is the supremum norm.
- So, it is very conservative and unforgiving.
- In ML, we often minimize an $L_{p}$[[p-Norm|-norm]] $e.g:$ $L_{2}$ 

> We can also provide a similar bound using a [[Stationary Distribution of Policy|stationary distribution]] of a [[Policy|policy]] $\pi$.

---
## See Also
- [[Approximate Value Iteration (AVI)]]
- [[Value Function Approximation]]
- [[Bellman Error]]