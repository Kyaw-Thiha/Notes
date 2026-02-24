# Regularized Loss Minimization
`Regularized Loss Minimization` (`RLM`) is the joint optimization of the [[Empirical Risk|empirical risk]] and the [[Classical ML/Regularization/Regularization|regularization]] function $\mathbb{R}: \mathbb{R}^d \to \mathbb{R}$

$$
\arg\min_{\mathbf{w}} (L_{S}(\mathbf{w}) + R(\mathbf{w}))
$$

The purposes of the `regularization function` is to penalizes a `hypothesis` for its effective complexity.

> [[Regularization]] gives us knobs to control the [[Bias-Variance|bias-variance trade-off]] and helps minimize the risk of overfitting.

---
## Commonly used p-Norm Regularizations
Here are different commonly used [[p-Norm]] regularizations.

$L_{1}\text{-norm}$: **Least Absolute Shrinkage and Selection Operator** (**LASSO**)

$$
\arg \min_{\mathbf{w} \in \mathbb{R}^d}
\left( \frac{1}{m} \sum^m_{i=1} \frac{1}{2}
(\mathbf{w} \cdot w_{i} - y_{i})^2 + \lambda ||\mathbf{w}||_{1} \right)
$$
Drives unnecessary weights to zero.
Will ignore highly correlated features.

---
$L_{2}\text{-norm}$: **Ridge Regression** (**Tikhonov Regularization**)

$$
\arg \min_{\mathbf{w} \in \mathbb{R}^d}
\left( \frac{1}{m} \sum^m_{i=1} \frac{1}{2}
(\mathbf{w} \cdot \mathbf{x}_{i} - y_{i})^2
+ \lambda ||\mathbf{w}||_{2}^2 \right)
$$
Works best when weights are of equal size.
Hence, feature preprocessing is like standardization.

---
$\text{Both}$: **Elasticnet**
$$
\arg\min_{\mathbf{w} \in \mathbb{R}^d}
\left( \frac{1}{m} \sum^m_{i=1} \frac{1}{2}
(\mathbf{w} \cdot \mathbf{x}_{i} - y_{i})^2 
+ \lambda_{2} ||\mathbf{w}||^2_{2} 
+ \lambda_{1} ||\mathbf{w}||_{1} \right)
$$

---

