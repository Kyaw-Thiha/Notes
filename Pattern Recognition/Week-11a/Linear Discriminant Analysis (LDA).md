# Linear Discriminant Analysis
#ml/classic-models/linear-discriminant-analysis

The assumptions we make to simplify the distribution here are:
1. Assume the labels are equally likely.
    $p(Y=0) = p(Y=1) = \frac{1}{2}$
2. $P(X\mid Y=0)$ and $P(X \mid Y=1)$ have the same covariance.
3. $P(X \mid Y=y)$ is a multivariate [[Gaussian Distribution]].

Then, our classifier is
$$
h_{Bayes}(\mathbf{x})
= \arg \max_{y=\{ 0,1 \}} p(Y=y) \ p(X=x \mid Y=y)
$$
will be equivalent to the statement
$$
\begin{align}
\log\left( \frac{p(Y=1) \ p(X=x \mid Y=1)} 
{P(Y=0) \ p(X=x \mid Y=0)} \right) &> 0 \\[10pt]

\implies \frac{1}{2}\left(  \ (x-\mu_{0})^{T} 
\ \Sigma^{-1} (x-\mu_{0}) \  \right)
- \frac{1}{2}\left(  \ (x-\mu_{1})^{T} 
\ \Sigma^{-1} (x-\mu_{1}) \  \right) &>0
\end{align}
$$
which we can rewrite as linear classifier $\boxed{ \ \text{step}(\mathbf{w}^{T} \mathbf{x} + b) \ }$ with
- $\mathbf{w} = (\mu_{1} - \mu_{0})^{T} \sum^{-1}$
- $b = \frac{1}{2}(\mu_{0} \Sigma^{-1}\mu_{0} - \mu_{1} \Sigma^{-1}\mu_{1})$

---
