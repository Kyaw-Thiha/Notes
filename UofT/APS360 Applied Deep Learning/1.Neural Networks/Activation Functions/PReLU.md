# PReLU(Parameterized Rectified Linear Unit) function
#### PReLU Activation Function
The [[PReLU|PReLU activation function]] is defined as
$$
f(x) = \begin{cases}
x, & \text{if } x \geq 0 \\
\alpha x, &\text{otherwise}
\end{cases}
$$
where $\alpha$ is **tuned** through [[Backpropagation|backpropagation]].

![|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/PReLU-Parametric-Rectified-Linear-Unit-Function.jpg)

Its main drawback is the brittleness of hyperparameter $\alpha$ which is different for different learning problems.

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