# Confidence Interval for Mean of  Normal Distribution
## Known Variance
Since this is a [[Gaussian Distribution (Normal Distribution)|normal distribution]], we know that 
$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{ n }} \sim \mathbb{N}(0,1)
$$
Let the [[Confidence Interval|confidence interval]] $\gamma = 0.95$. Then,
$$
\begin{align}
&P[k_{1} \leq z \leq k_{2}] = 0.95 \\[6pt]
\implies &P\left[ k_{1} \leq \frac{\bar{X} - \mu}{\sigma/\sqrt{ n }}  
\leq k_{2} \right] = 0.95 \\[6pt]

\implies &P\left[ k_{1} \frac{\sigma}{\sqrt{ n }} \leq \bar{X} - \mu  
\leq k_{2} \frac{\sigma}{\sqrt{ n }} \right] \\[6pt]

\implies &P\left[ k_{1} \frac{\sigma}{\sqrt{ n }} - \bar{X} \leq - \mu  
\leq k_{2} \frac{\sigma}{\sqrt{ n }} - \bar{X} \right] \\[6pt]

\implies &P\left[ \bar{X} - k_{2} \frac{\sigma}{\sqrt{ n }} \ \leq \mu  
\leq \ \bar{X} - k_{1} \frac{\sigma}{\sqrt{ n }} \right] \\[6pt]
\end{align}
$$
where
- $k_{2}$ is the lower interval
- $k_{1}$ is the upper interval

---
### Choice of $k_{1}$ and $k_{2}$ for any $\gamma$
The sampling distribution is unimodal and symmetric around the mode.
$$
z_{\left( \frac{1-\gamma}{2} \right)} \ \text{ and } \ z_{\left( \frac{1+\gamma}{2} \right)}
$$
are preferred values of $k_{1}$ and $k_{2}$ respectively.

**Example**: For $\gamma=0.95$, 
$$
\begin{cases}
k_{1} = z_{0.025} = -1.96 \\[6pt]
k_{2} = z_{0.975} = 1.96 \\
\end{cases}
$$

Hence for $X_{1}, X_{2}, \dots, X_{n} \overset{iid}{\sim} \mathbb{N}(\mu, \sigma^{2})$ with $\sigma^{2}$ known, we have the [[Confidence Interval|confidence interval]] $\gamma$ of mean $\mu$ as
$$
\left( \bar{X} - z_{\left( \frac{1+\gamma}{2} \right)} \frac{\sigma}{\sqrt{ n }}
\ , \quad  \bar{X} + z_{\left( \frac{1+\gamma}{2} \right)} 
\frac{\sigma}{\sqrt{ n }} \right) 
$$

---
### Example
**Question**: Let $(4.7, 5.5, 4.4, 3.3, 4.6, 5.3, 5.2, 4.8, 5.7, 5.3) \overset{iid}{\sim} \mathbb{N}(\mu, \sigma_{0}^{2})$ with $\sigma^{2}_{0}=0.5$.
Calculate the $0.95$-confidence interval for $\mu$.

**Solution**:
1. We have $n=10$, 
2. and a mean of
$$
\bar{X} = \frac{1}{10}(4.7 + 5.5 + \dots + 5.3) = 4.88
$$
3. Since $\gamma=0.95$, we have 
$$\frac{1+\gamma}{2} = 0.975$$
4. Using $z$-table, we get that 
$$
z_{0.975} \approx 1.96
$$
5. Hence the 0.95-CI for $\mu$ is
$$
4.88 \pm 1.96 * \frac{\sqrt{ 0.5 }}{\sqrt{ 10 }}
= (4.442, 5.318)
$$

---
## Unknown Variance
When $\sigma^{2}$ is unknown, we use $S^{2}$ as an [[Estimator|estimator]] of $\sigma^{2}$.
This means that we cannot use $\frac{\bar{X} - \mu}{\sigma/\sqrt{ n }} \sim \mathbb{N}(0,1)$ anymore, and instead use
$$
\frac{X-\mu}{S/\sqrt{ n }} \sim t_{(n-1)}
$$
Like [[#Choice of $k_{1}$ and $k_{2}$ for any $ gamma$|earlier]], for $X_{1}, X_{2}, \dots, X_{n} \overset{iid}{\sim} \mathbb{N}(\mu, \sigma^{2})$  with $\sigma^{2}$ unknown, we have $\gamma$-CI of $\mu$ as
$$
\left( \bar{X} - t_{\frac{1+\mu}{2} (n-1)} \frac{S}{\sqrt{ n }} 
\ , \quad \bar{X} + t_{\frac{1+\gamma}{2}(n-1)} \frac{S}{\sqrt{ n }} \right)
$$
where $t_{\frac{1+\gamma}{2}(n-1)}$ is the $\frac{1+\gamma}{2}$ quantile of a $t_{(n-1)}$ distribution.

---
### Example
**Question**: Let $(4.7, 5.5, 4.4, 3.3, 4.6, 5.3, 5.2, 4.8, 5.7, 5.3) \overset{iid}{\sim} \mathbb{N}(\mu, \sigma_{0}^{2})$ with both $\mu$ and $\sigma^{2}$ unknown.
Calculate the $0.95$-confidence interval for $\mu$.

**Solution**: 
1. We have $n=10$, 
2. and a mean of
$$
\bar{x} = \frac{1}{10}(4.7 + 5.5 + \dots + 5.3) = 4.88
$$
3. We first compute the [[Sample Variance|unbiased variance]] $S$ as
$$
\begin{align}
s = \sqrt{ \frac{1}{n-1} \sum(x_{i} - \bar{x})^{2}  } 
= \sqrt{ \frac{1}{n-1} \left( \sum x_{i}^{2} - n\cdot(\bar{x})^{2} \right) }
= 0.696
\end{align}
$$

4. Since $\gamma=0.95$, we have 
$$\frac{1+\gamma}{2} = 0.975$$
5. Using t-table or $R[qt(0.975, df=9)]$, we get
$$
t_{0.975(9)} \approx 2.262
$$
6. Hence the 0.95-CI for $\mu$ is
$$
4.88 \pm 2.262 * \frac{\sqrt{ 0.696 }}{\sqrt{ 10 }}
= (4.382, 5.378)
$$


---
