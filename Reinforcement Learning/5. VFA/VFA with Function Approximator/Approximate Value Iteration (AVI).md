# Approximate Value Iteration
#rl/vfa/approximate-value-iteration
Recall that [[Value Iteration]] is
$$
V_{k+1} \leftarrow TV_{k}
$$
with $T$ being either $T^{\pi}$ or $T^{*}$.

![Value Iteration|400](https://notes-media.kthiha.com/Approximate-Value-Iteration/4935951ae04a3e8a580b756c758f6e40.png)

One way to develop its approximate version is to perform each step only approximately.
$i.e:$ Find $V_{k+1} \in \mathcal{F}$ such that
$$
V_{k+1} \approx TV_{k}
$$

We start from a $V_{0} \in \mathcal{F}$.
Then at each iteration $k$ of [[Approximate Value Iteration (AVI)|AVI]], we solve
$$
V_{k+1} = \arg\min_{V \in \mathcal{F}}
||V - TV_{k}||_{p, \mu}
$$
The procedure for the [[Quality Function|action-value function]] is similar.

---
## Geometric View

When $\mathcal{F}$ is the set of linear functions, its geometry is the subspace spanned by $\phi$.

![AVI Geometric View|400](https://notes-media.kthiha.com/Approximate-Value-Iteration-(AVI)/b1f00a370ebc92179b73ed9c843418ed.png)

- Even though $V_{k} \in \mathcal{F}$, $TV_{k}$ may not be within $\mathcal{F}$ $(TV_{k} \in \mathcal{F})$.
  We visualize it with a point outside the plane.
- The amount of this error depends on
	- how expressive $\mathcal{F}$ is 
	- and, how much $T$ can push a function within $\mathcal{F}$ outside that space
- [[Value Iteration]] is a convergent procedure ([[Contraction of Bellman Operator|T is contraction]]).
  However, [[Approximate Value Iteration (AVI)|AVI]] may sometimes not converge, and it may indeed diverge.

----

