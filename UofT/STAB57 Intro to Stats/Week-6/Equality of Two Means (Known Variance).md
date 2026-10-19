# Equality of Two Means (Known Variance)
Suppose we have two independent normal distributions: 
- $X_{1}, \dots, X_{n} \sim N(\mu_{x}, \sigma^{2}_{x}) \implies \bar{X} \sim N\left( \mu_{x}, \ \frac{\sigma^{2}_{x}}{n} \right)$
- $Y_{1}, \dots, Y_{m} \sim N(\mu_{y}, \sigma^{2}_{y}) \implies \bar{Y} \sim N\left( \mu_{y}, \ \frac{\sigma^{2}_{y}}{m} \right)$

If the two means are equal, we would have
$$
\begin{align}
H_{0}: \
&\mu_{x} = \mu_{y} \\[6pt]
\implies &\mu_{x} - \mu_{y} = 0
\end{align}
$$
This distribution can be represented in 
$$
\boxed{ \ \bar{X} - \bar{Y} \ } \sim N\left( \boxed{ \ \mu_{x} - \mu_{y} \ }, 
\ \frac{\sigma_{x}^{2}}{n} + \frac{\sigma^{2}_{y}}{m} \right)
$$
where
- $\frac{\sigma_{x}^{2}}{n} + \frac{\sigma^{2}_{y}}{m}$ is based on the independence of distributions
- $\mu_{x} - \mu_{y}$ is the parameter being measured
- $\bar{X} - \bar{Y}$ is the [[Estimator|estimator]]

We can derive as
$$
\begin{align}
&\boxed{ \ \bar{X} - \bar{Y} \ } \sim N\left( \boxed{ \ \mu_{x} - \mu_{y} \ }, 
\ \frac{\sigma_{x}^{2}}{n} + \frac{\sigma^{2}_{y}}{m} \right) \\[6pt]
\implies &\frac{(\bar{X} - \bar{Y}) - (\mu_{x} - \mu_{y})}{ 
\sqrt{ \frac{\sigma^{2}_{x}}{n} + \frac{\sigma^{2}_{y}}{m} }} \sim \mathcal{N}(0,1)
& \text{by } (1)
\end{align}
$$
where
- $\frac{\sigma_{x}^{2}}{n} + \frac{\sigma^{2}_{y}}{m}$ is based on the independence of distributions
- $\mu_{x} - \mu_{y}$ is the parameter being measured
- $\bar{X} - \bar{Y}$ is the [[Estimator|estimator]]

Taking $H_{0}: \mu_{x} - \mu_{y} = 0$, we get that under $H_{0},$
$$
\begin{align}
\frac{\bar{X} - \bar{Y}}{\sqrt{ \frac{\sigma^{2}_{x}}{n} 
+ \frac{\sigma^{2}_{y}}{m} }} \sim \mathcal{N}(0, \ 1)
& & \text{ by } (2)
\end{align}
$$

---
## Approach-1
Using equation-$(1)$, we can construct a $(1-\alpha)$ confidence level of
$$
(\bar{X} - \bar{Y}) \pm z_{1-\frac{\alpha}{2}} \sqrt{ 
\frac{\sigma^{2}_{x}}{n} + \frac{\sigma^{2}_{y}}{m} }
$$
Since testing $H_{0}: \mu_{x} - \mu_{y} = 0$, check if $0$ is inside the interval or not.

---
## Approach-2
Using equation-$(2)$, we can construct the $\alpha$-level rejection region and calculate the test statistics or p-value.

---
## Special Case
> Testing $\mu_{x} = \mu_{y}$ with known $\sigma^{2}_{x} = \sigma^{2}_{y} = \sigma^{2}$.

We have that
- $X_{1}, \dots, X_{n} \sim \mathcal{N}(\mu_{x}, \ \sigma^{2}) \implies \bar{X} \sim \mathcal{N}\left( \mu_{x}, \ \frac{\sigma^{2}}{n} \right)$
- $Y_{1}, \dots, Y_{n} \sim \mathcal{N}(\mu_{y}, \ \sigma^{2}) \implies \bar{Y} \sim \mathcal{N}\left( \mu_{y}, \ \frac{\sigma^{2}}{n} \right)$

If the two means are equal, we would have
$$
\begin{align}
H_{0}: \
&\mu_{x} = \mu_{y} \\[6pt]
\implies &\mu_{x} - \mu_{y} = 0
\end{align}
$$

We get that
$$
\begin{align*}
& \overline{x} - \overline{y} \sim N\!\left(\mu_x - \mu_y,\; \frac{\sigma^2}{n} + \frac{\sigma^2}{m}\right) \\[6pt]

\implies & \frac{(\overline{x} - \overline{y}) - (\mu_x - \mu_y)}{\sqrt{\dfrac{\sigma^2}{n} + \dfrac{\sigma^2}{m}}} \sim N(0, 1) \\[6pt]

\implies & \frac{(\overline{x} - \overline{y}) - (\mu_x - \mu_y)}{\sigma\sqrt{\dfrac{1}{n} + \dfrac{1}{m}}} \sim N(0, 1)
\end{align*}
$$

---
## See Also 
- [[STAB57 Week-6]]
- [[Equality of Two Variances]]
- [[Equality of Two Means (Known Variance)]]
- [[Equality of Two Means (Unknown Variances)]]
- [[Comparing Two Population Means (Paired Data)]]
