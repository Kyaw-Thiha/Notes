# Understanding Maximum Likelihood and Risk Minimization
Previously, we have minimized the [[Negative Log Likelihood|negative log likelihood]] to fit the likelihood maximization into the [[Empirical Risk Minimization (ERM)|ERM framework]].

We know that using [[Gradient Descent|stochastic gradient descent]], we can minimize the true risk.
Expanding on that,
$$
\begin{align}
\mathbb{E}_{X}[\ell(\mathbf{w}, \ X)]
&= \mathbb{E}_{X}[-\log p(X; \ \mathbf{w})] \\[6pt]
&= - \sum_{\mathbf{x}} p(\mathbf{x}) \log \ 
p(\mathbf{x}; \ \mathbf{w}) \\[6pt]
&= \sum_{\mathbf{x}} p((\mathbf{x})  
\log\left( \frac{p(\mathbf{x})}{p(\mathbf{x}; \mathbf{w})} \right)
- \sum_{\mathbf{x}} p(\mathbf{x}) \log(p(\mathbf{x}))
\\[6pt]

&= D_{KL}(p(\mathbf{x}) \ || \ p(\mathbf{x}; \mathbf{w}))
+ H(p(x))
\end{align}
$$

The [[Empirical Risk|risk]] is minimized when the divergence between the true distribution and the estimated distribution is zero.
The trick becomes estimating the distribution.

---
## Algorithms
The following are algorithms are that make assumptions about the structure of the distribution in order to make computations simpler.
- [[Naive Bayes Review|Naive Bayes]]
- [[Linear Discriminant Analysis (LDA)]]
- [[Markov Models]]

---