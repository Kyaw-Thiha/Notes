# Leaky ReLU Activation Function
The [[Leaky ReLU]] function can be defined as
$$
f(x) = \max(cx, \ x)
$$
where $c \neq 0$ is a selected hyperparameter.

This allows non-zero output for negative values.

![|300](https://miro.medium.com/v2/resize:fit:1400/1*WeZjoLkFSjGCfJZkqSqZZA.png)

This non-horizontal sloped line for negative part allows [[Backpropagation|backpropagation]] for negative input values.

![|300](https://www.analytixlabs.co.in/wp-content/uploads/2023/08/Leaky-ReLU-derivative.jpg)

However the [[Gradient Descent Detail|gradient]] is still small, making learning to be slow. 

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