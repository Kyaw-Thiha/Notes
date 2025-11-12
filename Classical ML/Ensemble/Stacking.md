# Stacking
#ml/ensemble/stacking

`Stacking` is an [[Ensemble Model]] where you train multiple different `base models`, and then train another model - `meta-learner` - to combine their predictions optimally.

![Stacking](https://miro.medium.com/v2/0*woikIOVWv5vZMHpZ)

The idea is that different `base models` capture different aspects of the data:
- `Decision Trees` capture non-linear splits
- `Logistic Regression` capture global trends
- `k-NN` capture local structure

Then, the `meta-learner` learns the optimal way to trust each model in different regions of input space.

---

## Variants

#### Nested Stack

`Nested Stack` is a [[Stacking]] model where the `meta-learner` is also another [[Stacking]] model.

![Nested Stack](https://towardsdatascience.com/wp-content/uploads/2024/02/1BnoWy3Tn9to5y6uNTqSMdA.png)

#### Blending
Split the training set into `train set` ($80$%) and `holdout set`($20$%).
Train the `base models` on `train set`, and train the `meta-learner model` on `holdout set`

#### K-Fold Stacking
Just like [[K-Fold Cross Validation]], train each base model $h_{t}(x)$ on $K-1$ folds.
Then, predict on the held-out fold.
Finally, train the `meta-learner model` on these `out-of-fold` predictions.

#### Multi-Target Stacking
Each `base model` predicts one target $y_{k}$, and the `meta-learner model` learns to combines those outputs into one joint output $\hat{Y} = [y_{1}, \dots, y_{K}]$

#### Dynamic Stacking
Instead of just weighing each `base model`, we also weigh different values of $x$.
$$
F(x) = \sum_{k} w_{k}(x) \ h_{k}(x)
$$
where
- $h_{k}(x)$ is the `base model`
- $w_{k}(x)$ is the `weighing model`, usually `Neural Network`, or `Gating Network`.
This is used in `Mixture-of-Experts`, and `AutoML Systems`

#### Cross-Family Stacking
Use completely different `base models` -  `Decision Trees`, `CNN`, `LSTMs`, `linear models` - each capturing different structures.
Used in `Multi-Model Learning`, and `Tabular Data`

#### Feature Concatenation Stacking
Instead of just passing predictions from `base models`, also pass in the original features.
$$
X_{i} = [ h_{1}(x_{i}),. h_{2}(x_{i}), \dots, h_{K}(x_{i}), x_{i}]
$$
Useful when the `base models` are lossy learners.

---
## See Also
- [[Bagging]]
- [[Boosting]]
- [[Ensemble Model]]
