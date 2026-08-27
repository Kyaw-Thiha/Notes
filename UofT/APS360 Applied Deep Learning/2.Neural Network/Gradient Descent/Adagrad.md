# Adagrad
[[Adagrad]] adapts the learning rate to the parameters,
- performing smaller updates for parameters associated with frequently occurring features.
- and larger updates for parameters associated with infrequent features.

It is defined as
$$
\theta^{(t+1)}_{i}
= \theta^{(t)}_{i} - \frac{\eta}{\sqrt{ G^{(t)}_{ii}
+ \epsilon}} \cdot g^{(t)}_{i}
$$
where 
- $\theta^{(t)}_{i}$ is the parameter $i$ at time-step $t$
- $\eta$ is the learning rate
- $g^{(t)}_{i} = \nabla_{\theta}J(\theta^{(t)}_{i})$ is the partial derivative of [[Loss Function|objective function]]
- $G^{(t)} \in \mathbb{R}^{d \times d}$ where $G^{(t)}_{ii} = \sum^{t}_{\tau=1} g^{2}_{\tau,i}$, 
  is the diagonal matrix where each diagonal element $(i,i)$ is the sum of the squares of gradient $w.r.t$ $\theta_{i}$ up to time-step $t$

---
## Vectorization
For sake of brevity, let $g^{(t)}_{i}$ be the partial derivative of [[Loss Function|objective function]] $w.r.t$ $\theta_{i}$ at time step $t$.
$$
g^{(t)}_{i} = \nabla_{\theta}J(\theta^{(t)}_{i})
$$

**SGD Update**:
Recall that the [[Stochastic Gradient Descent(SGD)|SGD update]] for every parameters $\theta_{i}$ is
$$
\theta^{(t+1)}_{i} = \theta^{(t)}_{i} 
- \eta \cdot g^{(t)}_{i}
$$

**Adagrad**:
In its update rule, [[Adagrad|Adagrad]] modifies the learning rate $\eta$ based on past gradients that have been computed for $\theta_{i}$:
$$
\theta^{(t+1)}_{i}
= \theta^{(t)}_{i} - \frac{\eta}{\sqrt{ G^{(t)}_{ii}
+ \epsilon}} \cdot g^{(t)}_{i}
$$
where $G^{(t)}_{ii}$ is the running sum of squared gradients
$$
G^{(t)}_{ii} = \sum^{t}_{\tau=1} (g^{\tau}_{i})^{2}
$$
and is updated as
$$
G^{(t)}_{ii} = G^{(t-1)}_{ii} + (g^{(t)}_{i})^{2}
$$

**Vectorizing**:
We can thus vectorize this to get
$$
\theta^{(t+1)} = \theta^{(t)}
- \frac{\eta}{\sqrt{ G_{t} + \epsilon }}
\odot g_{t}
$$

---
## Benefits
> Eliminates the need to manually tune the learning rate. 
> Most implementations use default of $0.01$.

## Drawbacks
> Since every added term is positive, accumulated sum $G^{(t)}_{ii}$ keeps growing. This causes learning rate to shrink.

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
