# Risk Function
#ml/statistical-learning/risk-function

> The `risk` of a hypothesis $h$ is the `expected loss` when inputs are drawn from true data distribution $\mathcal{D}$ and labels are given by true target function $f$

Formally,
$$
L_{\mathcal{D},f}\ (h)
= \mathbb{E}_{x \sim \mathcal{D}} [ \ \mathcal{l} \ (h(x), f(x))\ ]
$$

It is also known as the `generalization error` or `true error` of a prediction rule $h:\mathcal{X} \to \mathcal{Y}$.

For a classifier, it is defined as
$$
\begin{align}
L_{D,f}(h)  
&\triangleq P_{x \sim \mathcal{D}} 
\ [ \ h(x) \neq f(x) \ ] \\[6pt]
&\triangleq \mathcal{D}( \ {x: h(x) \neq f(x)} \ )
\end{align}
$$
where
- $L_{D,f} \ (h)$ is the `risk` of hypothesis $h$ with respect to distribution $\mathcal{D}$ and target function $f$
- $\mathcal{D}$ is the `data distribution` 
- $f: \mathcal{X} \to \mathcal{Y}$ is the `true target labelling function`
- $h: \mathcal{X} \to \mathcal{Y}$ is the `hypothesis (predictor)`
- $\mathcal{D}(A)$ is the `probability mass` that distribution $\mathcal{D}$ assigns to the set $A$

---
`Probability Form`
$$
P_{x \sim \mathcal{D}} 
\ [ \ h(x) \neq f(x) \ ] 
$$
> Draw a random data $x$ from distribution $\mathcal{D}$
> It represents the probability that $h(x) \neq f(x)$  on that particular $x$

`Measure-of-set Form`
$$
\mathcal{D}( \ {x: h(x) \neq f(x)} \ )
$$
> Take the set of all points $x$ for which $h(x) \neq f(x)$ 
> Measure how much probability mass the distribution $\mathcal{D}$ assigns to that set

---
## Generalized Loss Function
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
## See Also
- [[Empirical Risk]]
- [[Empirical Risk Minimization (ERM)]]
