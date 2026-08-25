# Stochastic Gradient Descent
[[Stochastic Gradient Descent(SGD)|SGD]] essentially carries out [[Gradient Descent|gradient descent]] across individual data instead of over a batch. 

![300](https://miro.medium.com/1*Sa5kGcZIVNTLjrI8P-YsSQ.gif)

Its update rule is defined as
$$
\theta_{k+1} = \theta_{k} - \alpha_{k} \ g(\theta_{k}; \ \xi_{k})
$$
where
- $g(\theta_{k}; \xi_{k})$ is the [[Gradient Descent Detail|stochastic gradient]] of $F(x)$ at $\theta_{k}$
- $F(x) = E_{\xi \sim D} \ f(x; \xi)$ is the [[Risk Function|true risk]] we optimize
- $F_{S}(x) = \frac{1}{n}  \sum ^{n}_{i=1} f(x; \xi_{i})$ is the [[Empirical Risk|empirical risk]]
- $\mathbb{E}[g(\theta_{k}; \ \xi_{k})] = \nabla F(\theta_{k})$

---
## Better Convergence
[[Stochastic Gradient Descent(SGD)|SGD]] performs frequent updates with a high variance that cause the objective function to fluctuate heavily.

![300](https://storage.ghost.io/c/bd/89/bd89d480-b8e0-47e6-8bae-dcb0d061b8d7/content/images/2016/09/sgd_fluctuation.png)

[[Gradient Descent|Batch gradient descent]] converges to minimum basin the parameters are [[Gradient Descent Detail|initialized in]]. [[Stochastic Gradient Descent(SGD)|SGD]]'s fluctuation enable it to jump to new and potentially better local minima.

---
## Convergence Rate
This fluctuation can complicate convergence to exact minimum.

However when we slowly decrease the learning rate, [[Stochastic Gradient Descent(SGD)|SGD]] shows same convergence behaviour as [[Gradient Descent|batch gradient descent]]. It almost certainly converge to local or global minimum.

---
## Other Optimizers
- [[Gradient Descent]]
- [[Gradient Descent Detail]]
- [[Stochastic Gradient Descent(SGD)]]
- [[SGD with Momentum]]
- [[Nesterov Accelerated Gradient(NAG)]]
- [[Adagrad]]
- [[Adadelta]]
- [[RMS Prop]]
- [[Adaptive Moment Estimation(Adam)]]
- [[AdamW]]

---
## See Also
- [Good Blog by Sebastian Ruder](https://www.ruder.io/optimizing-gradient-descent/)
- [Blog covering SGD to latest methods](https://k4i.top/posts/optimizers-adamw/#from-update-to-sgd)
- [Paper with mathematical proofs of different optimizers](https://arxiv.org/pdf/2501.14458)