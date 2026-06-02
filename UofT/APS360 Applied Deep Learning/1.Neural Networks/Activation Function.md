# Activation Functions
[[Activation Function|Activation functions]] are mathematical equations attached to neurons in a [[neural network]] that determine whether a node should pass the signal forward.

![Activation Functions|400](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Activation-function-in-neural-network-1.jpg)

---
## Different Activation Functions

There are many different [[Activation Function|activation functions]] to consider.
We will start with more classical ones first.

![Different Activation Functions|500](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Types-of-activation-function.jpg)

---
### Linear Activation Function
Being the most trivial [[activation function]], it is also considered as an identity function, and is defined as
$$
f(x) = x
$$
for $y=f(w^{T}x + b)$.

![Linear Activation Function|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Linear-Activation-Function.jpg)

However, this has many problems such as
- only works for datasets that are [[Linear Predictor|linearly separable]]
- no advantage from deep [[Neural Network|linear layers]]

$$
\underbrace{W^{(1)} \times W^{(2)} \times \dots \times W^{(H)}}_{W'x}x
$$
- cannot apply [[Backpropagation|backpropagation]] since the derivative is constant

---
### Binary Step Functions
The first [[Neural Network|artificial neurons]]$(1943\text{ - }70s)$ used a [[Activation Function|binary activation function]] defined as
$$
f(x)
= \begin{cases}
0, & \text{if } x < 0 \\
1, & \text{if } x \geq 0 \\
\end{cases}
$$

Note that this function is neither differentiable, continuous, nor [[Lipschitz Smoothness|smooth]].
![|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Binary-Step-Function.jpg)

---
### Sigmoid Activation Function
[[Sigmoid Function|Sigmoid activation functions]] were the most common before $2012$.
They are 
- easily differentiable, [[Lipschitz Smoothness|smooth]], continuous
- range between $[-1, 1]$ or $[0,1]$

---
#### Logistic Function
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
#### Hyperbolic(Tanh) Function
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
### ReLU & Friends
#### ReLU Activation Function
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
#### Leaky ReLU
The [[Leaky ReLU]] function can be defined as
$$
f(x) = \max(cx, \ x)
$$
where $c \neq 0$ is a **selected** hyperparameter.

This allows non-zero output for negative values.

![|300](https://miro.medium.com/v2/resize:fit:1400/1*WeZjoLkFSjGCfJZkqSqZZA.png)

This non-horizontal sloped line for negative part allows [[Backpropagation|backpropagation]] for negative input values.

![|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Leaky-ReLU-derivative.jpg)

However the [[Gradient Descent Detail|gradient]] is still small, making learning to be slow. 

---
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
#### GELU
The [[GELU Function|Gaussian Error Linear Unit(GELU)]] is defined as
$$
\begin{align}
f(x)
&= x \ P(X \leq x)
= x \ \Phi(x) \\[6pt]
&= 0.5x \ \left( 1 + \tanh\left[ \sqrt{ \frac{2}{\pi} } (x + 0.044715x^{3}) \right] \right)
\end{align}
$$
This [[activation function]] combines
- [[ReLU]]
- [[Pattern Recognition/Week-10a/Dropout]]
- **Zoneout**: Randomly forces a few hidden units to maintain their prior value by mulitplying with $1$.

![|300](https://alaaalatif.github.io/gelu_imgs/gelu_viz-1.png)

The [[GELU Function]] combines all these properties so that the function randomly multiplies the input by $1$ or $0$ and gets output from [[Activation Function|activation function]] deterministically.

![GELU|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/GELU-Gaussian-Error-Linear-Unit-Function.jpg)

---
### Continuous Approximations of ReLU
#### ELU
The [[ELU Activation Function|exponential linear units(ELU)]] function is defined as
$$
\text{ELU}
= \begin{cases}
x & \text{for } x\geq 0 \\
\alpha(e^{x} - 1) & \text{for } x < 0
\end{cases}
$$
[[ELU Activation Function|ELU]] becomes gradually smooth until the output is same as $-\alpha$.

![|250](https://www.researchgate.net/publication/381125220/figure/fig4/AS:11431281263745902@1722304306240/Graphical-depiction-of-the-exponential-linear-unit-ELU-and-its-derivative-ELU.png)

The slope for the negative part is a log curve.
![|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/ELU-a1-Derivative.jpg)

Specifically, the derivative is
$$
f'(x)
= \begin{cases}
1 & \text{for } x\geq 0 \\
f(x) + \alpha & \text{for } x < 0
\end{cases}
$$

This log curve resolves the problem of dead neurons.
However, there are still a few issues like
- computational time due to exponential operation
- lack of learning for slope parameter $\alpha$
- exploding gradient problem

---
#### SiLU(Swish) Function
The [[SiLU(Swish) Function|SiLU function]] is defined as
$$
f(x) = x \cdot \sigma(x)
= \frac{x}{1 + e^{-x}}
$$
The [[SiLU(Swish) Function|swish function]] gradually bends from $0$ towards negative value, before moving upwards.
![|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Swish-Function.jpg)

Because of this bend, the [[SiLU(Swish) Function|Swish function]] can retain negative values for learning.

---
#### SoftPlus
The SoftPlus function is defined as
$$
\text{SoftPlus}(x)
= \frac{1}{\beta} \log(1 + e^{\beta x})
$$

---
### Other Functions
#### Softmax
The [[Softmax Function|softmax function]] is defined as
$$
\text{softmax}(x_{i})
= \frac{\exp(x_{i})}{\sum_{j} \exp(x_{j})}
$$

![|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Softmax-Function.jpg)

---
#### SELU
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
- [[Neural Network]]
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