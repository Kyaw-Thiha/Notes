# Nestrov Accelerated Gradient(NAG) Method

[[Nesterov Accelerated Gradient(NAG)|NAG]] is defined as 
$$
\begin{cases}
m_{k+1} &= \beta \ m_{k} + \alpha \ \nabla_{\theta}
J(\theta_{k} - \beta m_{k}) \\[6pt]

\theta_{k+1} &= \theta_{k} - m_{k+1}
\end{cases}
$$

---
## Intuition
[[SGD with Momentum|SGD with momentum]] can be thought of as a heavy-ball rolling down a hill. There, we want a smart ball that knows to slow down before the hill slopes up again. 

Recall that we use our momentum term $\beta m_{k}$ to move the parameters $\theta_{k}$. Computing $\theta_{k}- \beta \ m_{k-1}$ gives an approximation to next position of the parameters.

---
## Alternative Forms
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