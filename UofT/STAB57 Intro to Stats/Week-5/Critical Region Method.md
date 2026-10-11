# Critical Region
The [[Critical Region Method|critical region]] is a region of distribution of the [[#Test Statistics|test statistics]] such that we will reject [[Hypothesis Test|null hypothesis]] $H_{0}$ if $T(X) \in R_{\alpha}(T)$.

![|250](https://userpage.fu-berlin.de/soga/soga-py/200/2070_hypothesis_tests/img/Generalized_figure_of_two_sided_test.png)

We need to make sure that 
$$
P[ \ T(X) \in R_{\alpha}(T) \ |H_{0} \ \text{true}| \ ] = \alpha
$$

---
## Test Statistics
[[Test Statistics|Test statistics]] $T(X)$ is a quantity that
- sample data through an [[Estimator|estimator]]
- when $H_{0}$ is true, it has a known distribution
- under that distribution, it's possible to find some areas that has probability $\alpha$

The [[Pivotal Quanity|pivots]] used to construct [[Confidence Interval|confidence intervals]] are good examples of [[#Test Statistics|test statistics]]:
- $\frac{\bar{X}-\mu}{\sigma/\sqrt{ n }}$
- $\frac{\bar{X}-\mu}{S/\sqrt{ n }}$
- $\frac{(n-1)S^{2}}{\sigma^{2}}$

---
## Hypothesis Testing 
Let's [[Hypothesis Test|test the hypothesis]] $H_{0}: \mu = \mu_{0}$ when $X_{i} \overset{\text{iid}}{\sim} \mathbb{N}(\mu, \sigma^{2})$ when $\sigma^{2}$ is known.

![image|300](https://notes-media.kthiha.com/Critical-Region/a00d45775f47d86d383c6d3e5889bf7b.png)

We have the null hypothesis is
$$
H_{0}: \mu = \mu_{0}
$$

The [[Test Statistics|test statistics]] can be defined in 
$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{ n }}
$$
If $H_{0}$ is true $(\text{i.e: } \mu \neq \mu_{0})$, then 
$$
\frac{\bar{X} - \mu_{0}}{\sigma/\sqrt{ n }} \sim \mathbb{N}(0,1)
$$
The rejection area can be defined as
$$
\left( -\infty, z_{\frac{\alpha}{2}} \right) \cup
\left( z_{1-\frac{\alpha}{2}} , \infty \right)
$$
We then reject $H_{0}$ if 
$$
\frac{\bar{X} - \mu_{0}}{\sigma/\sqrt{ n }} < z_{\frac{\alpha}{2}}
\quad \text{or} \quad
\frac{\bar{X} - \mu_{0}}{\sigma/\sqrt{ n }} > z_{1 - \frac{\alpha}{2}}
$$

> **Intuition**
> We reject the [[Hypothesis Test|null hypothesis]] when [[Test Statistics|test statistics]] falls in the lower probability area of the distribution under the null.
> 
> If $\mu_{0}$ is the true mean, then $\bar{X}$ shouldn't be too far from $\mu_{0}$.

---
### Example-1
Let $(4.7, \ 5.5, \ 4.4, \ 3.3, \ 4.6, \ 5.3, \ 5.2, \ 4.8, \ 5.7, \ 5.3) \overset{iid}{\sim} \mathbb{N}(\mu, \ \sigma_{0}^{2})$ with $\sigma_{0}^{2} = 0.5$. 
Test $H_{0}:\mu=5$ at level of significance $\alpha=0.05$.

We first compute the mean to get
$$
\bar{x} = \frac{1}{10}(4.7 + 5.5 + \dots + 5.3) = 4.88
$$
We can then compute the [[Test Statistics|test statistics]] as
$$
T(X) = \frac{4.88-5}{\sqrt{ 0.5 }/\sqrt{ 10 }} = -0.537
$$
Given the level of significance $\alpha=0.05$, we have
$$
z_{\frac{0.05}{2}} \approx 1.96
$$
and thus, get the rejection region of
$$
(-\infty, \ -1.96) \cup (1.96, \ \infty)
$$
Since [[Test Statistics|test statistics]] value $-0.537$ does not fall in to the rejection area, we fail to reject $H_{0}$.

> **Note**: We never say we accept $H_{0}$.
> We failed to prove that $H_{0}$ is wrong $\not\implies$ $H_{0}$ is right.

---
### Example-2
Let $(4.7, \ 5.5, \ 4.4, \ 3.3, \ 4.6, \ 5.3, \ 5.2, \ 4.8, \ 5.7, \ 5.3) \overset{iid}{\sim} \mathbb{N}(\mu, \ \sigma_{0}^{2})$ with $\mu$ and $\sigma^{2}$ unknown.
Test $H_{0}:\mu=5$ at level of significance $\alpha=0.05$.

We first compute the mean to get
$$
\bar{x} = \frac{1}{10}(4.7 + 5.5 + \dots + 5.3) = 4.88
$$
and compute the [[Sample Variance|sample variance]] as
$$
s^{2} = \frac{1}{n-1} \sum^{n}_{i=1} (x_{i} - \bar{x})^{2} = 0.484
$$
We can then compute the [[Test Statistics|test statistics]] as
$$
T(X) = \frac{4.88-5}{\sqrt{ 0.484 }/\sqrt{ 10 }} = -0.545
$$
Given the level of significance $\alpha=0.05$, we have
$$
z_{\frac{0.05}{2}, \ 9} \approx 2.262
$$
and thus, get the rejection region of
$$
(-\infty, \ -2.262) \cup (2.262, \ \infty)
$$
Since [[Test Statistics|test statistics]] value $-0.545$ does not fall in to the rejection area, we fail to reject $H_{0}$.

---
