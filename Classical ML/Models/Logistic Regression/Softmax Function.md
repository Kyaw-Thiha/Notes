# Softmax Function
#ml/models/classic/logistic-regression/softmax  
#math/softmax

`Softmax Function` turns a vector of real numbers into a probability distribution.

$$
softmax(z_{i}) 
= \frac{e^{z_{i}}}{e^{z_{1}} + e^{z_{2}} + \dots + e^{z_{K}}} 
= \frac{e^{z_{i}}}{\sum^K_{j=1} e^{z_{j}}}
$$

![Softmax](https://miro.medium.com/v2/1*IYG4hy6twSajfumAhc1bnQ.gif)

---
## Multi-Class Cross Entropy
`Softmax Function` can be used to derive [[Cross-Entropy|Multi-Class Cross-Entropy]].

Let $P_{i,j} = \frac{e^{z_{j}}}{\sum^K_{l=1} e^{z_{l}}}$ where $z_{j} = w^T_{j} x_{i}$

Suppose that $y_{i}$ is encoded as `One-Hot Vector` 
$$
y_{i} = e_{j} = 
\begin{bmatrix} 
0 \\ \vdots \\ 1 \\ \vdots \\ 0 \end{bmatrix}
$$

Then, we can derive that
$$
\begin{align}
&L_{i}(w) = \begin{cases}
-\log P_{i,1} & y_{i} = 1 \\
-\log P_{i,2} & y_{i} = 2 \\
\vdots \\
-\log P_{i,K} & y_{i} = K
\end{cases} \\[6pt]

&\implies L_{i}(w) 
= -\sum^K_{j=1} y_{i}^T e_{j} \log P_{i,j}
\\[6pt]

&\implies E(w) 
= \sum^N_{i=1} L_{i}(w) 
= -\sum^N_{i=1} \sum^K_{j=1} y_{i}^T e_{j} \log P_{i,j}
\end{align}
$$

This means that we use can `Softmax Function` for [[Logistic Regression|Multi-Class Logistic Regression]].

---
## Numerical Issues

The `softmax function` $softmax(z_{j}) = \frac{e^{z_{j}}}{\sum^K_{l=1} e^{z_{l}}}$ has [[Floating Points|Numerical Issues]] of underflowing and overflowing.

Hence in practice, we use
$$
softmax(z_{j}) = \frac{e^{z_{j} - max(z)}}{\sum^K_{l=1} e^{z_{l} - max(z)}}
$$

---
## Relation to Sigmoid
Note that [[Sigmoid Function]] is a special case of `Softmax Function` used when it is a `Binary Classification`.

Suppose we have $2$ classes.
Then,
$$
\begin{align}
softmax(p_{1} = a)
&= \frac{e^a}{e^a - e^0} \\[6pt]
&= \frac{e^a}{e^a + 1} \\[6pt]
&= \frac{1}{1 + e^{-a}} \\[6pt]
&= \sigma(a) \\[6pt]
\end{align}
$$

---
## See Also
- [[Sigmoid Function]]
- [[Cross-Entropy]]
- [[Logistic Regression]]
