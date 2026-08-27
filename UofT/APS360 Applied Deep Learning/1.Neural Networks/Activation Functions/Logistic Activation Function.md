# Logistic Activation Function
The [[Logistic Regression|logistic function]] is defined as 
$$
f(x) = \frac{1}{1 + e^{-x}}
$$
![Sigmoid|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Sigmoid-Logistic-Function.jpg)

It has a derivative of
$$
f'(x) = \sigma(x)(1 - \sigma(x))
$$
![Sigmoid Derivative|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/logistic-activation-function.jpg)
Note that although it is differentiable, its gradient values are only significant between $-3$ and $3$.

This means values outside that range have small gradient.
This causes the `vanishing gradient` problem where the gradient value approaches $0$.

Thus, the [[Logistic Regression|logistic function]] is typically used in the output layers.

---
### ReLU Activation Function
Modern [[Neural Network|deep learning]] typically uses [[ReLU Activation Function|Rectified Linear Unit(ReLU)]], defined as
$$
\mathrm{ReLU}(x) = \max(0, x)
$$

It can be thought of as a piecewise linear function, but it has a derivative and allows for [[backpropagation]].

![|300](https://storage.ghost.io/c/3f/df/3fdf6ed2-17ac-4b12-a693-8078bd13e748/content/images/2023/06/relu-graph-1-1.jpeg)

Its advantages involve
- computational efficiency since neurons are deactivated when output $<0$.
- linear properties provide guarantee on [[Perceptron Convergence Theorem|convergence]] of the gradient descent towards [[Convex-Lipschitz-Bounded Learning Problem (CLB)|global minima]]

![|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/The-Dying-ReLU-problem.jpg)

Since the gradient becomes $0$, weights and biases cannot get updated during [[Backpropagation|backpropagation]]. This is called dying [[ReLU Activation Function|ReLU problem]].

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
