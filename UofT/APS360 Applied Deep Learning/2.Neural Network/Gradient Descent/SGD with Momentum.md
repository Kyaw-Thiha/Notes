# SGD with Momentum
[[SGD with Momentum|Momentum]] helps accelerate [[Stochastic Gradient Descent(SGD)|SGD]] in the relevant direction and dampens oscillations.

$$
\begin{align}
v_{k} &= \beta \ v_{k-1} + \nabla_{\theta} \  
J(\theta_{k-1}) \\[6pt]
\theta_{k+1} &= \theta_{k} - \alpha v_{k}
\end{align}
$$
where
- $\theta_{k}$: parameters at step $k$
- $\alpha$: learning rate
- $\beta$: momentum coefficient(typically $0.9$)
- $\nabla_{\theta} \ J(\theta_{k-1})$: gradient of loss $w.r.t$ parameters
- $v_{k}$: velocity

Intuitively $v_{k}$ can be thought of as an exponential moving average of gradients.

![image|300](https://notes-media.kthiha.com/SGD-with-Momentum/9994b224e73bf8643b2714174e99c2b2.png)

 Directions that remain consistent accumulate, while directions that keep flipping sign cancel out.

---
## Motivation
> **Ravine**/**Valley**: areas where curvature sharply inclines in one direction, often near the local optimal points. 

![image|300](https://notes-media.kthiha.com/SGD-with-Momentum/697518f55ec47de2f64571940b3bb75d.png)

- [[SGD with Momentum|SGD]] faces difficulties when moving through valleys in the optimization landscape. 
- It oscillate along the steep slopes of the valley. 
- This results in slow progress towards the local optimum at the bottom.

The introduction of the momentum term successfully addresses this issue by accelerating SGD in the pertinent
direction and mitigating oscillations.

---
## Momentum Methods
The difference between these methods lies in how momentum term $\beta \ m_{k}$ is used.

### Heavey-Ball(HB) Method
The stochastic heavey method is defined as
$$
\begin{cases}
m_{k+1} &= \beta \ m_{k} + \alpha \nabla g(\theta_{k}; \ \xi_{k}) \\[6pt]

\theta_{k+1} &= \theta_{k} - m_{k+1}
\end{cases}
$$
Note that this is same formula as above.

---
### Nestrov Accelerated Gradient(NAG) Method
The stochastic NAG method is defined as
$$
\begin{cases}
m_{k+1} &= \beta \ m_{k} + \alpha \nabla g(\theta_{k} - \beta m_{k}; \ \xi_{k}) \\[6pt]

\theta_{k+1} &= \theta_{k} - m_{k+1}
\end{cases}
$$
Note that this could also be written as
$$
\begin{cases}
y_{k} &= \theta_{k} + \beta(\theta_{k}  
- \theta_{k-1}) \\[6pt]
\theta_{k+1} &= y_{k} - \alpha \ \nabla_{\theta}  
J(y_{k})
\end{cases}
$$

It can also be approximated as 
$$
\begin{cases}
y_{k+1} &= \theta_{k} - \alpha \ \nabla_{\theta}  
J(\theta_{k}) \\[6pt]
\theta_{k+1} &= y_{k+1} + \beta(y_{k+1} - y_{k})
\end{cases}
$$
where $\nabla_{\theta} J(\theta_{k})$ does not need to be re-evaluated.

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
