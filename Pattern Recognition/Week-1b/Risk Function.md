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