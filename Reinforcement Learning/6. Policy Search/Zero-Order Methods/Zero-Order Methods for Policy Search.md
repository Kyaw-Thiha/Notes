# Zero-Order
[[Zero-Order Methods for Policy Search|Zero-Order]] only use value of objective at various query points.

They compute [[Policy Search Performance Measure|performance]] $J_{\rho}(\theta)$ at various $\theta$ in order to guide the optimization process.

---
## Methods
- If the [[Policy Parameterization|parameter space]] $\Theta$ is finite, use [[Finite Policy Parameter Space (Zero-Order)|these methods]].
- Else, use either [[Random Search (Zero-Order)|random search]] or [[Evolutionary Algorithms RL|evolutionary algorithms]].

---