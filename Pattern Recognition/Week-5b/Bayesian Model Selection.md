# Bayesian Model Selection
Consider a set of hypotheses (models)
$$
\{ h_{i} \}, \text{ for } i=1, \dots, L
$$
that lets us estimate probabilities of the samples $(x_{i}, y_{i})$ in the training set $S$.

We can then estimate the posterior distribution as
$$
p(h_{i} \mid S)
\propto p(S \mid h_{i}) \ p(h_{i})
$$
where
- $p(S \mid h_{i})$ is the `model evidence` or `marginal likelihood`
- $p(h_{i})$ is the `prior belief` in the hypothesis $h_{i}$

---
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

