# P-value 
[[P-Value|p-Value]] is the smallest level of significance at which $H_{0}$ would be rejected based on the observed data. 

Its the probability of observing the result as or more extreme than that actually observed if [[Hypothesis Test|hypothesis]] $H_{0}$ is true.


> If [[P-Value|p-value]] is less than a predefined cut-off, we reject $H_{0}$.

Conventionally, we compare [[P-Value|p-value]] to $0.01$, $0.05$ or $0.1$.

---
## Z-Test
For the [[#Z-Test|z-test]], we use
$$
2\left[ 1 - \Phi\left( |\frac{\bar{x} - \mu_{0}}{\sigma/\sqrt{ n }}| \right) \right]
$$
where $\Phi$ is the CDF of a [[Gaussian Distribution (Normal Distribution)|standard normal distribution]].

---
### Example
Let $(4.7, \ 5.5, \ 4.4, \ 3.3, \ 4.6, \ 5.3, \ 5.2, \ 4.8, \ 5.7, \ 5.3) \overset{iid}{\sim} \mathbb{N}(\mu, \ \sigma_{0}^{2})$ with $\sigma_{0}^{2} = 0.5$. 
Test $H_{0}:\mu=5$ at level of significance $\alpha=0.05$.

### Solution
From [[Critical Region Method#Example-1|critical method example-1]], we had our test statistics of $T(x)=-0.537$.
Computing the [[P-Value|p-value]], we get
$$
2\left[ 1 - \Phi\left( |\frac{\bar{x} - \mu_{0}}{\sigma/\sqrt{ n }}| \right) \right]
= 2(1 - \text{pnorm}(0.537)) \approx 0.5912
$$

---
## T-Test
For the [[#T-Test|t-test]], we use
$$
2\left[ 1 - G\left( | \frac{\bar{x}-\mu_{0}}{s/\sqrt{ n }} | \right) \right]
$$
where $G$ is the CDF of a $t_{(n-1)}$ distribution.

---
### Example-2
Let $(4.7, \ 5.5, \ 4.4, \ 3.3, \ 4.6, \ 5.3, \ 5.2, \ 4.8, \ 5.7, \ 5.3) \overset{iid}{\sim} \mathbb{N}(\mu, \ \sigma_{0}^{2})$ with $\mu$ and $\sigma^{2}$ unknown.
Test $H_{0}:\mu=5$ at level of significance $\alpha=0.05$.

### Solution
From [[Critical Region Method#Example-2|critical method example-2]], we had our test statistics of $T(x)=-0.545$.
Computing the [[P-Value|p-value]], we get
$$
2\left[ 1 - G\left( | \frac{\bar{x}-\mu_{0}}{s/\sqrt{ n }} | \right) \right]
= 2(1 - \text{pt}(0.545, \ df=9)) \approx 0.5989
$$

---
