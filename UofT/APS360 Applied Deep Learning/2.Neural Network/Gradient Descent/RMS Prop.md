# RMS Prop
[[RMS Prop]] seeks to reduce the monotonically decreasing learning rate of [[Adagrad]]. It is defined as
$$
\begin{align}
\theta_{t+1} &= \theta_{t} - \frac{\eta}{\sqrt{ v_{t} + \epsilon }} \odot g_{t}
\end{align}
$$
where
- $v_{t} = \gamma \ v_{t-1} + (1 - \gamma) \ g^{2}_{t}$  is [[Adadelta|running average of squared gradients ]]
- $g^{(t)}_{i} = \nabla_{\theta}J(\theta^{(t)}_{i})$ is the partial derivative of [[Loss Function|objective function]]
- $\eta$ is the [[Learning Rate|learning rate]]

> It can be thought of as [[Adadelta]] without the unit mismatch fix.

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