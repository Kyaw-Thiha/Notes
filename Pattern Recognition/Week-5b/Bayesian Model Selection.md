# Bayesian Model Selection
Consider a set of hypotheses (models)
$$
\{ h_{i} \}, \text{ for } i=1, \dots, L
$$
that lets us estimate probabilities of the samples $(x_{i}, y_{i})$ in the training set $S$.

We can then estimate the posterior distribution as
$$
p(h_{i} \mid S)
\ \propto \ p(S \mid h_{i}) \ p(h_{i})
$$
where
- $p(S \mid h_{i})$ is the `model evidence` or `marginal likelihood`
- $p(h_{i})$ is the `prior belief` in the hypothesis $h_{i}$

---
## Bayes Factor
To compare two models, we consider the `Bayes factor` $K$
$$
K = \frac{p(S \mid h_{i})}{p(S \mid h_{j})}
$$

The `Bayes factor` tells us how many more times likely $h_{i}$ is than $h_{j}$.
- $K > 1$ implies $h_{i}$ is more likely.
- $K < 1$ implies $h_{j}$ is more likely.

For finite $S$, `Bayes Factor` can be tricked.
But on average $w.r.t$ $S$, it will pick the right hypothesis.

To select a model, choose the model with the largest $p(h_{i} \mid S)$.

---
## Bayesian Information Criterion
Pick a model that minimizes the `Bayesian Information Criterion (BIC)`:
$$
\text{BIC}
= k \ln(m) - 2\ln( \ p(S \mid h_{i}) \ )
$$
where
- $k$ is the no. of free parameters in the model
- $m$ is the training set size

> As $m \to \infty$, `BIC` approximates model selection using Bayes Factor.


**Benefits**
Note that `BIC`
- doesn't require using a prior distribution
- explicitly penalizes model complexity (free parameters)

**Drawbacks**
But `BIC`
- is only valid when $m \gg k$
- cannot handle complex collections of models

---
## Prediction with Model Posterior
Model selection requires picking one best model.
But not maintaining a belief about other model is not very `Bayesian`.

In order to maintain all the hypotheses instead of picking just one, we can average models using the posterior distribution:
$$
p(y \mid \mathbf{x}, S)
= \sum^L_{i=1} p(y \mid \mathbf{x}, h_{i}, S)
\ \ p(h_{i} \mid S)
$$
This of course comes with the cost of having to maintain all the models.
However depending on hypothesis space, this maybe compact.

---