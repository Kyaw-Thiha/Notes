# SELU
The [[SELU Function|Scaled Exponential Linear Unit(SELU)]] function is defined as
$$
f(\alpha, x)
= \lambda \begin{cases}
\alpha(e^{x} - 1) & \text{for } x < 0 \\
x & \text{for } x \geq 0
\end{cases}
$$
[[SELU Function|SELU]] performs initial normalization by preserving the mean and variance of each layer from previous layers.

![|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/SELU-Scaled-Exponential-Linear-Unit-Function.jpg)

The variable in [[SELU Function|SELU]] is adjusted using [[Gradient Descent|gradients]].
As external normalization is slower than internal normalization, [[SELU Function|SELU]] converges quickly.

---
## See Also
- [Good Article](https://www.analytixlabs.co.in/blog/activation-function-in-neural-network/)
- [[Activation Function]]
- [[Logistic Activation Function]]
- [[Tanh Activation Function]]
- [[ReLU Activation Function]]
- [[Leaky ReLU]]
- [[PReLU]]
- [[ELU Activation Function]]
- [[SiLU(Swish) Function]]
- [[GELU Function]]
- [[Softmax Function]]