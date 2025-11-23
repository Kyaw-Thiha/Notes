# Learning Logistic Regression
#ml/models/classic/logistic-regression  

`Definitions`
Recall from [[Logistic Regression]] that 
$$
P(c_{i}|x) = \frac{1}{1 + e^{-w^Tx}}
$$

Let
- $N$ be the no. of training points.
- $D$ be the dimension of input features.
- $X = \{ x_{i} \}_{i=1}^N$ be the input data.
- $Y = \{ y_{i} \}_{i=1}^N$ be the target data.

`Goal`: Select $w^*$ that maximizes the $P(Y|X, \ w)$

---
`Initial Derivation`
$$
\begin{align}
P(X, Y \ | \ w)
&= P(Y|X,w) \ P(X|w) \\[6pt]
&= k.P(Y|X,w)  
& \text{since } X \text{ does not depends on } w  
\\[6pt]
&\propto P(Y | X,w) \\[6pt]
&= \Pi^N_{i=1} P(y_{i} \ | \ x_{i}, w)
\end{align}
$$

`Binary Classification`
Suppose we are doing `Binary Classification` such that
- $P_{i} = P(y=c_{1} | x_{i}, \ w) = g(w^T x_{i}) = \frac{1}{1+e^{-w^Tx_{i}}}$
- $1 - P_{i} = P(y=c_{0} | x_{i}, w)$

Then,
$$
\begin{align}
&\Pi^N_{i=1} P(y_{i} \ | \ x_{i}, w) \\[6pt]
&= \Pi_{i:y_{i}=1} P_{i} \ \Pi_{i:y_{i}=0} (1-P_{i})
&\text{by independence} \\[6pt]
&= \Pi^N_{i=1} P_{i}^{y_{i}} \ (1-P_{i})^{(1-y_{i})}
\end{align}
$$

`MLE Likelihood`
To maximize the likelihood, we need to apply [[Log Likelihood|Negative Log Likelihood]]  
$$
w_{MLE} = \arg \max_{w} P(Y|X,w) = \arg \min_{w} 
-\log P(Y|X,w)
$$
Applying it, we get
$$
\begin{align}
E(w)
&= -\log \Pi^N_{i=1} P_{i}^{y_{i}} (1-P_{i})^{1-y_{i}} \\[6pt]

&= - \sum^N_{i=1} [y_{i} \log P_{i} + (1-y_{i}) \log(1-P_{i})] \\[6pt]

\end{align}
$$
Note that this is same idea as [[Cross-Entropy|Binary Cross Entropy Loss]].

By $P_{i} = g(w^T x_{i}) = g(a(w))$, we get
$$
\begin{align}
&- \sum^N_{i=1} [y_{i} \log P_{i} + (1-y_{i}) \log(1-P_{i})] \\[6pt]

&= -\sum^N_{i=1} [y_{i} \ \log(g(w^T x_{i})) +  
(1- y_{i}) \log(1 - g(w^T x_{i}))] \\[6pt]

&= -\sum^N_{i=1} [y_{i} \ \log(g(a(w))) +  
(1- y_{i}) \log(1 - g(a(w)))] \\[6pt]
\end{align}
$$

`Differentiation`
Recall from [[Sigmoid Function]] that
- $\frac{\partial}{\partial w} \log g(a(w)) = (1- g(a)) \ x_{i}$
- $\frac{\partial}{\partial w} \log [1 - g(a(w))] = - g(a) \ x_{i}$

Using it, we can derive that
$$
\begin{align}
\nabla E(w)
&= -\sum^N_{i=1} [y_{i} \  (1- g(a)).x_{i} - (1-y_{i}) \ g(a).x_{i}] \\[6pt]
&= -\sum^N_{i=1} [y_{i}x_{i} - y_{i}g(a)x_{i} - g(a)x_{i} + y_{i}g(a)x_{i}] \\[6pt]
&= -\sum^N_{i=1} (y_{i}x_{i} - g(a).x_{i}) \\[6pt]
&= -\sum^N_{i=1} (y_{i}x_{i} - P_{i}.x_{i}) \\[6pt]
&= -\sum^N_{i=1} (y_{i}x_{i} - g(w^T x_{i}).x_{i}) \\[6pt]
&= -\sum^N_{i=1} (y_{i} - g(w^T x_{i})) \ x_{i} \\[6pt]
\end{align}
$$

Note that there is not closed-form solution for $\nabla E(w) = 0$.
Hence, we need to solve it by [[Iterative Refinement Algorithm|Numerical Methods]] like [[Gradient Descent]] or [[Newton's Method]].

---
## See More
- [[Logistic Regression]]
- [[Sigmoid Function]]
- [[Cross-Entropy]]
- [[Gradient Descent]]
- [[Newton's Method]]
