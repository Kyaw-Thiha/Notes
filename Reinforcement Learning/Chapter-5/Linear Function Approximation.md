# Linear Function Approximation
#rl/vfa/linear-function-approximation

The `Linear Function Approximator` is defined based on a set of basis functions:
$$
\hat{V}(s) 
= \phi(s)^{T} w
= \sum^{p}_{i=1} \phi_{i}(s) \ w_{i}
$$
with $w \in \mathbb{R}^{p}$ and $\phi: \mathcal{S} \to \mathbb{R}^{d}$.

Any $\hat{V}$ belongs to the space of functions $\mathcal{F}$ defined as
$$
\mathcal{F}
= \left\{ x \mapsto \phi(s)^{T}w : 
w \in \mathbb{R}^{p} \right\}
$$
This function space $\mathcal{F}$ is called the `value function space`.

In this example, it is a span of a set of features. 
We simply call it a linear function space.

> Note that the linearity is in the parameters w and not in the state x.

---