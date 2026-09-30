# Confidence Interval for Population Proportion
## Bernoulli Distribution
Let $X_{1}, X_{2}, \dots, X_{n} \sim \text{Bern}(\theta)$ with $\theta = P[X_{i}=1]$.

$(1)$: We know that by properties of [[Bernoulli Distribution|Bernoulli distribution]], 
- $\mathbb{E}[X_{i}] = \theta$
- $V[X_{i}] = \theta(1-\theta)$

$(2)$: Recall from [[Central Limit Theorem(CLT)]] that for an arbitrary distribution
$$
\begin{align}
&X_{1}, \dots, X_{n} \sim ?(\mu, \sigma^{2}) \\[6pt]
\implies &\bar{X} \to \mathbb{N}\left( \mu, \ \frac{\sigma^{2}}{n} \right)
\end{align}
$$
Using [[Bernoulli Distribution]] and [[Central Limit Theorem(CLT)]], we get
$$
\bar{X} \overset{D}{\to} \mathbb{N}\left( \theta, \ \frac{\theta(1-\theta)}{n} \right)
$$
For $\gamma$-level [[Confidence Interval|CI]] for $\theta$, we have
$$
\bar{X} \pm z_{\frac{1+\gamma}{2}} \cdot \sqrt{ \frac{\theta(1-\theta)}{n} }
$$
Since this is not directly usable, we can replace $\theta$ by $\bar{X}$ to get
$$
\bar{X} \pm z_{\frac{1+\gamma}{2}} \cdot \sqrt{ \frac{\bar{X}(1-\bar{X})}{n} }
$$

---
### Example
**Question**: Suppose $\theta$ represents the proportion of Covid-19 cases in UofT.
We randomly selected $200$ students and $15$ of them tested positive for Covid.
Calculate a $0.95$ confidence interval for $\theta$.

**Solution**: First, we calculate the mean
$$
\bar{x} = \frac{15}{200} = 0.075
$$
Next, 
$$
z_{\frac{1+\gamma}{2}} = z_{0.975} = 1.96
$$
Computing the [[Confidence Interval|confidence interval]], 
$$
\begin{align}
&0.075 \pm 1.96 \sqrt{ \frac{0.075(1-0.075)}{200} } \\[6pt]
= \ &0.075 \pm 0.0365 \\[6pt]
= \ &(0.0385, \ 0.111)
\end{align}
$$

---
## Poisson Distribution
Let $X_{1}, X_{2}, \dots, X_{n} \sim \text{Pois}(\theta)$ where $X_{i} = \# \text{ of accidents on 401 on any given day}$.

We know that by properties of Poisson Distribution, 
- $\mathbb{E}[X_{i}] = \lambda = \mu$
- $V[X_{i}] = \lambda = \sigma^{2}$
- Estimator$=\bar{X}$

By [[Central Limit Theorem(CLT)]], we have
$$
\bar{X} \to \mathbb{N}\left( \lambda, \frac{\lambda}{n} \right)
$$
For $\gamma$-level [[Confidence Interval|CI]] for $\theta$, we have
$$
\bar{X} \pm z_{\frac{1+\gamma}{2}} \cdot \sqrt{ \frac{\lambda}{n} }
$$

Since this is not directly usable, we can replace $\theta$ by $\bar{X}$ to get
$$
\bar{X} \pm z_{\frac{1+\gamma}{2}} \cdot \sqrt{ \frac{\bar{X}}{n} }
$$

---
