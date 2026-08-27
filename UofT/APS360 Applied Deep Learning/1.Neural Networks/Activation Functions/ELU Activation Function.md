# ELU Activation Function
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