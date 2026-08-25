# Adaptive Moment Estimation(Adam)
[[Adaptive Moment Estimation(Adam)|Adam]] is defined as
$$
\theta_{t+1} = \theta_{t} - 
\frac{\eta}{\sqrt{ \hat{v}_{t} + \epsilon }}
\ \hat{m}_{t}
$$
where
- $m_{t} = \beta_{1} \ m_{t-1} + (1 - \beta_{1})\  g_{t}$ is decaying average of `gradients`, and is also known as `momentum`
- $v_{t} = \beta_{2} \ v_{t-1} + (1 - \beta_{2})\  g_{t}^{2}$ is decaying average of `squared gradients`
- $\hat{m}_{t} = \frac{m_{t}}{1 - \beta_{1}^{t}}$ is the bias-corrected first moment
- $\hat{v}_{t} = \frac{v_{t}}{1 - \beta_{2}^{t}}$ is the bias-corrected second moment

---
## Formulation
#### Running Average of Gradients
In addition to storing an exponentially decaying average of past squared gradients $v_{t}$ like [[Adadelta]] and [[RMS Prop]], it also keeps an exponentially decaying average of past gradients $m_{t}$

$$
\begin{align}
m_{t} &= \beta_{1} \ m_{t-1} + (1 - \beta_{1})\  g_{t} \\[6pt]
v_{t} &= \beta_{2} \ v_{t-1} + (1 - \beta_{2})\  g_{t}^{2}
\end{align}
$$

which can be thought of as
- $m_{t}$ being the estimate of the first moment(mean)
- $v_{t}$ being the estimate of the second moment(uncentered variance)

#### Bias Correction
As $m_{t}$ and $v_{t}$ are initialized as $0$, they are biased towards $0$.
To correct these biases, we compute
$$
\begin{align}
\hat{m}_{t} &= \frac{m_{t}}{1 - \beta_{1}^{t}} \\[6pt]
\hat{v}_{t} &= \frac{v_{t}}{1 - \beta_{2}^{t}}
\end{align}
$$

#### Parameter Update
We update the parameters similar to [[Adadelta]] and [[RMS Prop]]:
$$
\theta_{t+1} = \theta_{t} - 
\frac{\eta}{\sqrt{ \hat{v}_{t} + \epsilon }}
\ \hat{m}_{t}
$$
with default values of
- $\beta_{1} = 0.9$
- $\beta_{2} = 0.999$
- $\epsilon = 10^{-8}$

---
## Benefits
Its incorporation of momentum and adaptive learning rate means that it 
- has rapid convergence
- require minimal tuning
- and is the commonly used optimizer

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