# Gradient Descent with Non-Linearities
We may have a nonlinearity on our predictor.
Think of [[Logistic Regression|logistic regression]] or perhaps a [[Neural Network|neural activation function]] like `ReLU`.

Then,
$$
h(\mathbf{x}) = \sigma(\mathbf{w} \cdot \mathbf{x})
$$

Consequently, when we compute the gradient of loss, we get
$$
\begin{align}
&\nabla \ell(\mathbf{w}, \ (\mathbf{x}, \mathbf{y}))
\\[6pt]

&= \ell'(\mathbf{w}, \ (\mathbf{x}, \mathbf{y}))
\ \nabla_{\varphi}(\mathbf{w} \cdot \mathbf{x})  
\\[6pt]

&= \ell'(\mathbf{w}, \ (\mathbf{x}, \mathbf{y}))
\ \varphi'(\mathbf{w} \cdot \mathbf{x})  
\ \nabla(\mathbf{w} \cdot \mathbf{x}) \\[6pt]

&= \ell'(\mathbf{w}, \ (\mathbf{x}, \mathbf{y}))
\ \varphi'(\mathbf{w} \cdot \mathbf{x})  
\ \mathbf{x} \\[6pt]
\end{align}
$$

---
## Examples
### Least Squares and ReLU
Remember that $\text{ReLU}(x) = \max\{ 0,x \}$.
We can define a [[Subgradient|subgradient]] for this `activation function` as well.

$$
\begin{align}
&\nabla \ell(\mathbf{w}) \\[6pt]

&= \nabla(\text{ReLU}(\mathbf{w} \cdot \mathbf{x}) 
- \mathbf{y})^{2} \\[6pt]

&= \nabla(\text{ReLU}(\mathbf{w} \cdot \mathbf{x}) 
- \mathbf{y}) \ \nabla 
\text{ReLU}(\mathbf{w} \cdot \mathbf{x}) \\[6pt]

&= \nabla(\text{ReLU}(\mathbf{w} \cdot \mathbf{x}) 
- \mathbf{y}) 
\begin{cases}
0 & \text{if } \mathbf{w} \cdot \mathbf{x} \leq 0  \\
1 & \text{if } \mathbf{w} \cdot \mathbf{x} > 0
\end{cases} \\[6pt]

&= \nabla(\text{ReLU}(\mathbf{w} \cdot \mathbf{x}) 
- \mathbf{y})  
\ \mathbb{1}(\mathbf{w} \cdot \mathbf{x} > 0)   
\ \mathbf{x}
\end{align}
$$

---
### Negative Log Likelihood and Multiclass Logistic Regression