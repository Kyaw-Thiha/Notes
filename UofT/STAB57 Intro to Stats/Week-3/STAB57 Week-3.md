# Week-3
## Inference about $\theta$ (Prev Week)
For point estimation, we have
- [[Method of Moments Estimation|Methods of Moments]] $( \ \theta = \text{func of } \mathbb{E}[X], \ \mathbb{E}[X^{2}], \ \dots \ )$
- [[Maximum Likelihood Estimation(MLE)|MLE]] $\left(  \ L(\theta) = \prod^{n}_{i=1} f(x_{i}) \  \right)$, which we can find with
	- $\frac{dL(\theta)}{d\theta} = 0$
	- analytically in the case of $x \geq \theta$ or $\theta \leq x$

The common estimator is $\bar{X}$:
- $X_{i} \sim \mathbb{N}(\mu, \ \sigma^{2}) \implies \bar{X} \sim \mathbb{N}\left( \mu, \frac{\sigma^{2}}{n} \right)$
- $X_{i} \sim \mathbf{?}(\mu, \ \sigma^{2}) \implies \bar{X} \overset{D}{\to} \mathbb{N}\left( \mu, \frac{\sigma^{2}}{n} \right)$

---
## Content
### Quality of Estimator
- To measure the quality of the [[Estimator|estimator]], we can use [[Mean Squared Error(MSE)]]. 
- [[Sample Variance]]

---
### Sample Distribution of Unbiased Variance
- [[Independence between Mean and Variance]] ($\bar{X}$ and $S^{2}$ are independent)
- [[Chi-Squared Distribution of the Sample Variance]] ($\frac{(n-1)S^{2}}{\sigma^{2}} \sim \mathcal{X}^{2}_{(df = n-1)}$)

---
## Exercises
- [[STAB57 Week-3 Exercise]]
- [[STAB57 Week-3 Tutorial]]

---