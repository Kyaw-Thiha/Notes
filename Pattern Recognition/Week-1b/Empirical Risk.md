# Empirical Risk
#ml/statistical-learning/empirical-risk

The `empirical risk` can be defined as
$$
L_{S}(h) \triangleq
\frac{1}{M} \sum^M_{m=1} \mathbb{1}(h(x_{m}) \neq y_{m})
$$
where
- $S  = \{ (x_{1}, y_{1}), \ \dots,  \ (x_{m}, y_{m}) \}$ is the `sample`(dataset)
- $M = |S|$ is the `dataset size`
- $h(x_{m})$ is the `prediction` of hypothesis on $x_{m}$ input
- $y_{m}$ is the `true label` of $m^{th}$ example

This can also be considered as `training error`, since this is the error classifier incurs over the training sample.

> `Note`
> We have been able to define this learning paradigm without discussing the learning algorithm $A(S)$ or the expression of hypothesis $h$

`Empirical Risk Minimization (ERM)`
The process of finding $h$ that minimizes $L_{S}(h)$ is called `ERM`.

---
## Empirical Risk with Loss Function
A [[Loss Function]] can be defined as
$$
\mathcal{l} 
= \mathcal{H} \times \mathcal{X}
\times \mathcal{Y}
\to \mathbb{R}_{+}
$$
For sake of compactness, let $\mathcal{Z} = \mathcal{X} \times \mathcal{Y}$. 
Then,
$$
\mathcal{l} 
= \mathcal{H} \times \mathcal{Z}
\to \mathbb{R}_{+}
$$
Using this, we can re-define the [[Risk Function|True Risk Function]] as
$$
L_{\mathcal{D}}(h)
\triangleq \mathbb{E}_{z \sim \mathcal{D}} 
[\mathcal{l}(h, z)]
$$
Likewise, the [[Empirical Risk Minimization (ERM)|Empirical Risk]] can be re-defined as
$$
L_{S}(h)
\triangleq \frac{1}{m}
\sum^m_{i=1} \mathcal{l}(h, z_{i})
$$
---
