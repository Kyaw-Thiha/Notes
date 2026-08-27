# Hyperbolic(Tanh) Function
The hyperbolic tangent function can be defined as
$$
f(x) = \frac{(e^{x} - e^{-x})}{(e^{x} + e^{-x})}
$$

Although it has a similar `S` shape like the [[Logistic Activation Function|logistic activation function]], its output is $[-1,1]$ instead of $[0,1]$.

![|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Tanh-Hyperbolic-Tangent-Function.jpg)

This zero-centered design allows the network to center the data.

![|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Tanh-derivative.jpg)

It still faces the same vanishing gradient problem as the [[Sigmoid Function|sigmoid function]]. However, tanh is preferred over [[Sigmoid Function|sigmoid]] because it is zero-centered.

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
