# GELU Function
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
- [[Dropout]]
- **Zoneout**: Randomly forces a few hidden units to maintain their prior value by mulitplying with $1$.

![|300](https://alaaalatif.github.io/gelu_imgs/gelu_viz-1.png)

The [[GELU Function]] combines all these properties so that the function randomly multiplies the input by $1$ or $0$ and gets output from [[Activation Function|activation function]] deterministically.

![GELU|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/GELU-Gaussian-Error-Linear-Unit-Function.jpg)

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