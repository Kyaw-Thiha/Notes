# Evolutionary Algorithms
> Solution to optimization problem is an individual in population.
> The [[Value Function|value of function]] is optimized as fitness of that individual.

Emulate the evolution:
- Mutation
- Reproduction
- Selection

There are different variants towards approaching this:
- Genetic Algorithms
- Genetic Programming
- Evolutionary Strategy

---
## Evolutionary Strategy
![image|500](https://notes-media.kthiha.com/Evolutionary-Algorithms-RL/e742f6e5783b3ba069b95005c7f71150.png)

[[#Evolutionary Strategy|Evolutionary Strategy(1+1)]] is similar to [[Random Search (Zero-Order)|Random Search]] but the choice of randomness over $\pi$ is guided.

A modification of this algorithm is called $\text{ES}(1, \lambda)$ with $\lambda > 1$.
- The parent $\theta_{k}$ generates $\lambda$ offsprings.
$$
\theta'_{k,j}
= \theta_{k} + \sigma_{k} \eta_{j}
\ , \quad j = 1, \ \dots, \ \lambda
$$
- The competition would only be between the offsprings $\{ \theta'_{k,j} \}^{\lambda}_{j=1}$ and not with the parent.
- Only one of the $\lambda$ offsprings gets to the next generation.

---
## See Also
- [[Finite Policy Parameter Space (Zero-Order)]]
- [[Random Search (Zero-Order)]]
- [[Evolutionary Algorithms RL]]
- [[Policy Search Performance Measure]]
- [[Policy Parameterization]]

