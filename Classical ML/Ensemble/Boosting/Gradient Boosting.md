# Gradient Boosting
#ml/ensemble/boosting/adaboost

`Gradient Boosting` is a variant of [[Boosting]] that fits each model to `negative gradient` of the [[Loss Function]].

![Gradient Boosting](https://media.geeksforgeeks.org/wp-content/uploads/20250903173429506712/des.webp)

---
## Model Definition

We model the [[Ensemble Model]] $F(x)$ as additive expansion of base models $h_{k}(x)$
$$
F(x) = F_{0}(x) + \sum^K_{i=1} \eta \ \alpha_{k} \ h_{k}(x)
$$
where 
- $F_{0}(x)$ is the initial prediction (mean of $y$ for `regression`)
- $h_{k}(x)$ is the `base model` at $k$
- $\eta \in (0, 1]$ is the learning rate
- $\alpha_{k}$ is the scaling factor

---
## Algorithm

First, train the first model.
$$
F_{0}(x) = arg \min_{\gamma} \sum_{i} L(y_{i}, \gamma)
$$
 Note that for `MSE`, this will give use mean of $y_{i}$.

Second, compute the `pseudo-radicals`.
The `pseudo-radicals` are `negative gradients` of the [[Loss Function]] with respect to current `prediction`.
$$
r_{i}^{(k)}
= - \left[ \frac{\partial \ L(y_{i}, F(x_{i}))} {F(x_{i})} \right]
$$

To allow our model to generalize based on `test dataset`, fit the base model $h_{k}(x)$ to predict `radicals` by fitting on `pseudo radicals`  $x_{i} \to r_{i}^{(k)}$ 

Third, compute the step size
$$
\alpha_{k} = \arg \min_{\alpha} \sum_{i} L(y_{i}, \ F_{k-1}(x_{i}) + \alpha \ h_{k}(x))
$$
This can be solved analytically (for `MSE`), or by `line search`.

Fourth, update the model by
$$
F_{k}(x) = F_{k-1}(x) + \eta \ \alpha_{k} \ h_{k}(x)
$$
where
- $\eta \in (0, 1]$ is the `learning rate` (usually $0.05$ or $0.1$)
- $\alpha_{k}$ is the `step size`
- $h_{k}(x)$ is the `residual prediction model`

---

## See Also
- [[AdaBoost]]
- [[Ensemble Model]]
