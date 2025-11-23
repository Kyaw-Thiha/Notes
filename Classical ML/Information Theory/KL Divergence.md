# KL Divergence
#ml/information-theory/kl-divergence  

`Kullback-Leibler (KL) Divergence`, also known as `Relative Entropy`, measures the difference between the two distributions.
$$
D_{KL}(P \ || \ Q) = \sum_{i=1}^N P(x_{i}).\log\left( \frac{P(x_{i})} {Q(x_{i})} \right)
$$

![KL-Divergence](https://bekaykang.github.io/assets/img/post/2022-01-15.gif)

---
`Relation to Cross-Entropy`
Given two distributions $Q(x)$ and $P(x)$, the `relative entropy` of $Q$ with respect to $P$ is
$$
\begin{align}
D_{KL}(Q(x) \ || \ P(x))

&= \sum_{i=1}^N  Q(x_{i}) \log\left(  \frac{Q(x_{i})}{P(x_{i})}  \right)  \\[6pt]

&= \sum_{i=1}^N Q(x_{i}) \log(Q(x_{i}))  
- \sum_{i=1}^N Q(x_{i}) \log(P(x_{i})) \\[6pt]

&= -H(Q) \underbrace{- E_{Q}[\log P(x_{i})]}_{cross-entropy} \\[6pt]

&= -H(Q) + \underbrace{H(Q, \ P)}_{cross-entropy}
\end{align}
$$

In other words, the [[Entropy]] of $Q$ is
$$
H(Q) = H(Q, P) - D_{KL}(Q \ || \ P)
$$

[[Cross-Entropy|Read More about Cross-Entropy here]]

---
`Lack of Commutativity`
Note that
$$
D_{KL}(Q \ || \ P) \neq D_{KL} (P \ || \ Q)
$$
since $H(Q, P) \neq H(P, Q)$.

---
## See More

- [[Cross-Entropy]]
- [[Entropy]]
- [[Math Behind VAE]]