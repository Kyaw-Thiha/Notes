Trivial Baseline: Global Mean
$R^2$ measures how well the model perform against the trivial baseline.

$$
R^2 = 1 - \frac{\sum^N_{i=1} (y_{i} - \hat{y}_{i})^2}{\sum^N_{i=1} (y_{i} - \tilde{y}_{i})^2}
$$

where
$$
\tilde{y} = \frac{1}{K}.\sum^N_{i=1}y_{i}
$$

- $R^2 = 1$: Maximum value (sign of overfitting) ($K$ is too small)
- $R^2 = 0$: No better than baseline ($K$ is too large)
- $R^2 < 0$: Worse than global mean baseline