# AdamW
[[AdamW]] is defined as
$$
\theta_{t+1} = \theta_{t} - \eta \left(
\frac{\hat{m}_{t}}{\sqrt{ \hat{v}_{t} 
+ \epsilon }} + \lambda \theta_{t} \right)
$$
where
- $m_{t} = \beta_{1} \ m_{t-1} + (1 - \beta_{1})\  g_{t}$ is decaying average of `gradients`, and is also known as `momentum`
- $v_{t} = \beta_{2} \ v_{t-1} + (1 - \beta_{2})\  g_{t}^{2}$ is decaying average of `squared gradients`
- $\hat{m}_{t} = \frac{m_{t}}{1 - \beta_{1}^{t}}$ is the bias-corrected first moment
- $\hat{v}_{t} = \frac{v_{t}}{1 - \beta_{2}^{t}}$ is the bias-corrected second moment
- $\lambda \theta_{t}$ is the weight decay with decay rate $\lambda$

> It can be thought of as [[Adaptive Moment Estimation(Adam)|Adam]] but with decoupled [[Weight Decay|weight decay]].

---
## Motivation
If we were to add [[Weight Decay|weight decay]] to [[Adaptive Moment Estimation(Adam)|Adam]] naively,
$$
g_{t} = \nabla J(\theta_{t}) + \lambda \theta_{t}
$$
This modified $g_{t}$ would flow through the entire [[Adaptive Moment Estimation(Adam)|Adam]] pipeline, thus causing non-uniform decay per parameter since [[Adaptive Moment Estimation(Adam)|Adam]] is adaptive to each parameter.

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
