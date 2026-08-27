# Adadelta
[[Adadelta]] seeks to reduce the monotonically decreasing learning rate of [[Adagrad]]. It is defined as
$$
\Delta \theta_{t}
= - \frac{\sqrt{ u_{t-1} + \epsilon }}
{\sqrt{ v_{t} + \epsilon }} \ g_{t}
\quad , \quad 
\theta_{t+1} = \theta_{t} + \Delta \theta_{t}
$$
where
- $v_{t} = \gamma \ v_{t-1} + (1 - \gamma) \ g^{2}_{t}$ is running average of `squared gradients`
- $u_{t} = \gamma \ u_{t-1} + (1 - \gamma) \ \Delta \theta^{2}_{t}$ is the running average of `squared parameter updates`

---
## Formulation
#### Running Average of Squared Gradients
First lets define the decaying average of all past squared gradients as 
$$
v_{t} = \gamma \ v_{t-1} + (1 - \gamma) \ g^{2}_{t}
$$
where $\gamma$ is set to similar value as momentum term of $0.9$.

---
#### Parameter Update Vector
Lets define the parameter update vector $\Delta \theta_{t}$ as
$$
\begin{align}
\Delta \theta_{t} &= - \eta \cdot g_{t,i} \\[6pt]
\theta_{t+1} &= \theta_{t} + \Delta \theta_{t}
\end{align}
$$
The parameter update vector of [[Adagrad]] is thus
$$
\Delta \theta_{t} = - \frac{\eta}{\sqrt{ G_{t} + \epsilon }}
\odot g_{t}
$$
In [[Adadelta]], we replace the diagonal matrix of running sums $G_{t}$ with running average $v_{t}$:
$$
\Delta \theta_{t} = - \frac{\eta}{\sqrt{ v_{t} + \epsilon }} \odot g_{t}
$$

---
#### Running Average of Squared Parameter Updates
It is noted that units in this parameter update (as well as [[Stochastic Gradient Descent(SGD)|SGD]], [[SGD with Momentum|Momentum]], or [[Adagrad|Adagrad]]) do not match. To fix this, let us define a decaying average of squared parameter updates:
$$
u_{t} = \gamma \ u_{t-1} + (1 - \gamma) \ \Delta \theta^{2}_{t}
$$

---
#### Defining Adadelta
This lets us then define [[Adadelta]] as 
$$
\Delta \theta_{t}
= - \frac{\sqrt{ u_{t-1} + \epsilon }}
{\sqrt{ v_{t} + \epsilon }} \ g_{t}
\quad , \quad 
\theta_{t+1} = \theta_{t} + \Delta \theta_{t}
$$

Note that this means we don't need the learning rate $\eta$ as a [[Hyperparameter|hyperparameter]] in [[Adadelta]].

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