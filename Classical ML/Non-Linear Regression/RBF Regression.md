# RBF Regression
#ml/classic-models/non-linear-regression/rbf

`RBF Regression` is a type of [[Non-Linear Regression]] that uses `Radial Basis Functions` to fit the data.

![RBF Regression](https://gamedevacademy.org/wp-content/uploads/2017/10/Curve-Fitting.gif)

This is how a `RBF Regression` model looks like in 2 dimensions.
![[RBF Regression.png]]

`RBF Regression` is good at capturing local relations.

For each `RBF`, we can control 2 parameters:
- `Width` $\sigma$: Determines how wide/narrow each function is
- `Center` $c$: Determines the position of the center of each function

| Spacing ($\Delta$) | Width ($\sigma$) | Overlap   | Behavior                | Risk                   |
| ------------------ | ---------------- | --------- | ----------------------- | ---------------------- |
| Small              | Small            | Very low  | Highly localized, spiky | Overfit                |
| Small              | Large            | Very high | Redundant, smooth       | Underfit / redundancy  |
| Large              | Small            | Gappy     | Missing regions         | Underfit               |
| Large              | Large            | Moderate  | Smooth coverage         | Stable (if tuned well) |

## Choosing width & centers
For choosing `Width` $\sigma$ and `Center` $c$,
- `Low-Dimensional Data`
  Centers $c$ on uniform grid, and width $\sigma$ proportional to grid spacing.
- `High-Dimensional Data`
  Choose centers $c$ via [[K-Means|K-Means clustering]] and width $\sigma$ proportional to average inter-center distance
- `Heuristic`
  $\sigma = \alpha \times \text{mean distance between centers}$, where $c \in [0.5, 1.5]$
- `Grid Search`
  Minimize test/validation error over a grid of ($\sigma$, $M$)

## See Also
- [[Polynomial Regression]]
- [[Non-Linear Regression]]
