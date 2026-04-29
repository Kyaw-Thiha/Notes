# Value Iteration
#rl/planning/value-iteration 
`Value iteration` iteratively improve the estimates of the value of each state.

![Value Iteration|300](https://notes-media.kthiha.com/Value-Iteration/baf24f021ed440cc4d128eda1bc7fbd4.png)

---
## Value Iteration for Policy Evaluation
Starting from $V_{0} \in \mathcal{B}(\mathcal{S})$, we compute a sequence of $(V_{k})_{k\geq0}$ by
$$
\begin{align}
&V_{k+1} \leftarrow T^{\pi}V_{k} \\[6pt]
\implies &V_{k+1} 
\leftarrow r^{\pi} + \gamma \ \mathcal{P}^{\pi} V_{k}
\end{align}
$$
By [[Uniqueness of Fixed Point]] of the [[Bellman Operator]], 
$$
\lim_{ k \to \infty } 
|| V_{k} - V^{\pi} ||_{\infty} = 0
$$
meaning for any $s \in \mathcal{S}$, we have $V_{k}(s) \to V^{\pi}(s)$.

> The [[Value Function|value function]] $V_{k}$ or [[Quality Function|quality function]] $Q_{k}$ generated through `VI` procedure have corresponding [[Greedy Policy|greedy policies]] $\pi_{g}$.
> By this [[Greedy Policy of Optimal Value Function is Optimal Policy|theorem]], as the value functions converge to optimal value functions, their corresponding greedy policies also converge to an `optimal policy`.

---
## Value Iteration for Control
By [[Uniqueness of Fixed Point]] of [[Bellman Optimality Operator]],
$$
\begin{align}
&V_{k+1} \leftarrow T^{*} V_{k} \\[6pt]
&Q_{k+1} \leftarrow T^{*} Q_{k} \\[6pt]
\end{align}
$$
it is guaranteed that $V_{k} \to V^{*}$ (or $Q_{k} \to Q^{*}$).

![Value Iteration|400](https://notes-media.kthiha.com/Value-Iteration/42ca8414ae0f797aaa0d0641e9c2e939.png)

> **Bootstrapping**
> The `VI` uses an existing approximation $V_{k}$ of $V^{*}$ to get get a better approximation of $V^{*}$.
> This idea is called `bootstrapping` in the RL literature.

---
## See Also
- [[Value Function]]
- [[Uniqueness of Fixed Point]]
- [[Greedy Policy of Optimal Value Function is Optimal Policy]]
