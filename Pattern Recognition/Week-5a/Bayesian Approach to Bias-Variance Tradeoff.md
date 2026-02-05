# Bayesian Approach to Bias-Variance Tradeoff
Recall that the [[Empirical Risk]] of our predictor is
$$
L_{S}(h_{S})
= \mathbb{E}_{S}[ \ (y - h_{S}(x))^2 \ ]
$$

This can be considered as the `variance` of the predictions.

---
Let's consider [[Bayesian Linear Regression]], where the predictive distribution over observations is
$$
p(y \mid \mathbf{x}, \mathbf{y}, \alpha, \beta)
= \mathcal{N}(y \mid \mu_{m}^T \varphi(\mathbf{x})
, \ \sigma_{m}^2(\mathbf{x}))
$$
The `variance` of this prediction $\sigma^2_{m}(\mathbf{x})$ is given by
$$
\sigma^2_{m}(\mathbf{x})
= \frac{1}{\beta} + \varphi(\mathbf{x})
\ \mathbf{S}_{m} \varphi(\mathbf{x})
$$
where $\beta$ is the `noise term` across observations.

It has been proven that $\sigma^2_{m+1}(\mathbf{x}) \leq \sigma^2_{m}(\mathbf{x})$.
Hence as number of observations $m \to \infty$, the only term remaining in the prediction variance is the `noise parameter` $\frac{1}{\beta}$.

---
## See Also
- [[Bias-Variance]]
- [[Bayesian Linear Regression]]