# Fast Convergence of Policy Iteration
#rl/planning/policy-iteration

It can be shown that the [[Policy Iteration|Policy Iteration algorithm]]  converges in 
$$
O\left( \frac{|\mathcal{S}| |\mathcal{A}|}
{1 - \gamma} \cdot \log\left( \frac{1}{1- \gamma} \right) \right)
$$
iterations.

This is improvement over the result in [[Convergence of Policy Iteration Algorithm]].

---
## Proof
The proof for this can be found in [Fundamentals of RL by Farahmand, Page 86-92](https://amfarahmand.github.io/IntroRL/lectures/FRL.pdf)

---
## See Also
- [[Policy Iteration]]
- [[Convergence of Policy Iteration Algorithm]]
- [[Policy Improvement Theorem]]