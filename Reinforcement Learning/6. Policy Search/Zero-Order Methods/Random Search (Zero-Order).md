# Random Search
- Randomly pick $m$ [[Policy Parameterization|policy parameters]] $\theta_{1}, \ \dots, \ \theta_{m} \in \Theta$.
- Evaluate [[Policy Search Performance Measure|performance]] $\hat{J}_{n}(\pi_{\theta_{i}})$.
- Pick the one with the highest value.

With large enough $m$, one of $\theta_{i}$ might hit close to [[Policy|optimal policy]]
$$
\hat{\theta}
\leftarrow \arg\max_{\theta \in \Theta}
\hat{J}_{n}(\pi_{\theta})
$$
If $n$ is large enough, the difference between $\hat{J}_{n}(\theta)$ and $J_{\rho}(\theta)$ would be small for all randomly selected $\theta$.

---
## Pseudocode
![image|500](https://notes-media.kthiha.com/Random-Search-(Zero-Order)/201100c18326c3a41ce98c899bb512ae.png)

---
## Notes
- Can provide guarantee that [[Random Search (Zero-Order)]] finds the optimal point, asymptotically.
- [[Random Search (Zero-Order)|RS]] is not the most efficient way to search [[Policy Parameterization|parameter space]].
- Note that it does not benefit from all evaluations of $\hat{J}_{n}$, when suggesting a new $\theta'_{k}$.
- Instead of blindly sampling from same distribution $v$, we can adaptively change the [[policy]] parameter sampling distribution $v_{k}$ to be a function of prior evaluations.

---
## See Also
- [[Finite Policy Parameter Space (Zero-Order)]]
- [[Random Search (Zero-Order)]]
- [[Evolutionary Algorithms RL]]
- [[Policy Search Performance Measure]]
- [[Policy Parameterization]]