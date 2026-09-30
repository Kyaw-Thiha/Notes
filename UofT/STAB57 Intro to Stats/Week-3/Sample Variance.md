# Sample Variance
We can define [[Sample Variance|population variance]] $\sigma^{2}$ as
$$
\sigma^{2} = \mathbb{E}[(X - \mu)^{2}]
$$
where $\mu = \mathbb{E}[X]$.

---
## Estimators
If we have equally likely $N$ data points in our population, this is equivalent to 
$$
\hat{\sigma}^{2} = \frac{1}{n} \sum^{n}_{i=1}(x_{i} - \bar{x})^{2}
$$
or similarly as an unbiased estimator of
$$
C^{2} = \frac{1}{n-1} \sum^{n}_{i=1}(x_{i} - \bar{x})^{2}
$$
Note that in statistical literature, [[Sample Variance|sample variance]] always refers to the unbiased estimator.

---
## Checking Unbiasedness
### Property to Check Unbiasedness
$$
\begin{align}
\sum(x_{i} - \mu)^{2}
&= \sum ^{n}_{i=1} (\underbrace{x_{i} - \bar{x}}_{a}  
+ \underbrace{\bar{x} - \mu}_{b})^{2} \\[6pt]
&= \sum^{n}_{i=1}(x_{i} - \bar{x})^{2} + \sum^{n}_{i=1}(\bar{x} - \mu)^{2}
+ \sum^{n}_{i=1} 2(x_{i} - \bar{x})\underbrace{(\bar{x} - \mu)}_{\text{const}}
\\[6pt]
&= \sum^{n}_{i=1}(x_{i} - \bar{x})^{2} + n(\bar{x} - \mu)^{2}
+ 2(\bar{x} - \mu) \sum^{n}_{i=1} \underbrace{(x_{i} - \bar{x})}_{0} \\[6pt]
&= \sum^{n}_{i=1}(x_{i} - \bar{x})^{2} + n(\bar{x} - \mu)^{2}
\end{align}
$$
Hence, we get that
$$
\sum(x_{i} - \bar{x})^{2} = \sum^{n}_{i=1}(x_{i} - \mu)^{2} 
+ n(\bar{x} - \mu)^{2}
$$

---
### Checking for unbiasedness of $C^{2}$
Taking $E[ \ \ ]$ on both sides,
$$
\begin{align}
E\left[ \sum(x_{i} - \bar{x}) \right]
&= E\left[ \sum(x_{i} - \mu)^{2} \right] - E[n(\bar{x} - \mu)^{2}] \\[6pt]
&= \sum E[(x_{i} - \mu)^{2}] - n E[(x - \mu)^{2}] \\[6pt]
&= \sum V[x_{i}] - n E[(\bar{x} - E[\bar{x}])^{2}] \\[6pt]
&= \sum V[x_{i}] - n V[\bar{x}] \\[6pt]
&= \sum \sigma^{2} - n \frac{\sigma^{2}}{n} \\[6pt]
&= n \sigma^{2} - \sigma^{2} \\[6pt]
&= \sigma^{2} (n-1) \\[6pt]
\end{align}
$$

Now, divide both sides by $(n-1)$, 
$$
\begin{align}
&\frac{E\left[ \sum(x_{i} - \bar{x})^{2} \right]}{n-1} = \sigma^{2} \\[6pt]
\implies &E\left[ \frac{1}{n-1} \sum (x_{i} - \bar{x})^{2} \right] = \sigma^{2}
\\[6pt]
\implies &E[S^{2}] = \sigma^{2}
\end{align}
$$
Hence, $S^{2}$ is unbiased. 

---
### Checking for biasedness of $\hat{\sigma}^{2}$
Alternatively, if we divide by $n$, we have
$$
E\left[ \frac{1}{n} \sum(x_{i} - \bar{x}) \right] = \frac{n-1}{n}\sigma^{2}
\neq \sigma^{2}
$$
$\therefore$ $\hat{\sigma}^{2}$ is biased.

To get how much bias it has, we can do
$$
\begin{align}
\text{Bias}[\hat{\sigma}^{2}]
&= E[\hat{\sigma}^{2}] - \sigma^{2} \\[6pt]
&= \frac{n-1}{n} \sigma^{2} - \sigma^{2} \\[6pt]
&= \sigma^{2} \left( \frac{n-1-n}{n} \right) \\[6pt]
&= \sigma^{2}\left( -\frac{1}{n} \right)
\end{align}
$$
Note that when $n\to \infty$, $\text{Bias}[\hat{\sigma}^{2}] \to 0$.

---
## Unbiasedness of Sample Variance using Chi-Sq Distribution
Recall that the mean of a [[Chi-Squared Distribution]] is its degrees of freedom $df$.
Then since [[Sample Variance|standardized sample variance]] $\frac{(n-1)S^{2}}{\sigma^{2}} \sim \mathcal{X}^{2}_{(df=n-1)}$
$$
\begin{align}
\mathbb{E}\left[ \frac{(n-1)S^{2}}{\sigma^{2}} \right] &= (n-1) \\[6pt]
\mathbb{E}[S^{2}] &= \sigma^{2}
\end{align}
$$

This proves that $S^{2}$ is an [[Estimator|unbiased estimator]] for $\sigma^{2}$ under [[Gaussian Distribution (Normal Distribution)|normal distribution]].

---