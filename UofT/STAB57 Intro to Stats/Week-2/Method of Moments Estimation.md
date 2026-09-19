# Method of Moments Estimation
Let $X_{1}, \ X_{2}, \ \dots, \ X_{n}$ are independently and identically distributed$(i.i.d)$ random variables. Let the $k^{th}$ [[Moment Generating Function(MGF)|population moment]] be
$$
\mu_{k} = \mathbb{E}[X^{k}]
$$
Then, the $k^{th}$ [[Moment Generating Function(MGF)|sample moment]] based on sample is
$$
\hat{\mu}_{k} = \frac{1}{n} \sum^{n}_{i=1} X^{k}_{i}
$$
We use $\hat{\mu}_{k}$ as an [[Estimator|estimator]] of $\mu_{k}$.

---
### Example-1
**Question**: Let $X_{1}, \ X_{2}, \ \dots, \ X_{n} \overset{\text{iid}}{\sim}  \text{Poisson}(\lambda)$. Find the [[Method of Moments Estimation|method of moments estimator]] of $\lambda$.

**Solution**: Since, its a Poisson Distribution,
$$
\lambda = E[x]
$$
Then, compute to get
$$
\hat{\lambda} = \frac{1}{n} \sum X_{i} = \bar{X}
$$

---
### Example-2
**Question**: Let $X_{1}, \ X_{2}, \ \dots, \ X_{n} \overset{\text{iid}}{\sim}  \text{N}(\mu, \sigma^{2})$. Find the [[Method of Moments Estimation|method of moments estimators]] of $\mu$ and $\sigma^{2}$.

**Solution**: We need to find the following parameters:
$$
\begin{align}
&\mu = \mathbb{E}[X] & \quad
& \sigma^{2} = \mathbb{E}[X^{2}] - (\mathbb{E}[X])^{2}
\end{align}
$$
To estimate $\mu=\mathbb{E}[X]$, we use
$$
\hat{\mu} = \frac{1}{n} \sum x_{i} = \bar{x} \\[6pt]
$$
and to estimate $\sigma^{2} = \mathbb{E}[X^{2}] - (\mathbb{E}[X])^{2}$, we use
$$
\begin{align}
\hat{\sigma}^{2}
&= \frac{1}{n} \sum x_{i}^{2} - (\bar{x}^{2})
\\[6pt]
&= \frac{1}{n} \sum(x_{i} - \bar{x})^{2} &\text{by (1)}
\end{align}
$$

**Proof of** $(1)$: Starting from result, we have
$$
\begin{align}
&\frac{1}{n} \sum(x_{i} - \bar{x})^{2} \\[6pt]

\implies & \frac{1}{n} \sum(x_{i}^{2}  
- 2x_{i}\bar{x} + \bar{x}^{2}) \\[6pt]

\implies & \frac{1}{n} \sum x_{i}^{2}
- \frac{1}{n} \sum 2x_{i} \bar{x}
+ \frac{1}{n} \sum \bar{x}^{2} \\[6pt]

\implies & \frac{1}{n} \sum x_{i}^{2}
- \frac{2x_{i}}{n} \sum \bar{x}
+ \frac{\bar{x}^{2}}{n} \sum 1 \\[6pt]

\implies & \frac{1}{n} \sum x_{i}^{2}
- \frac{2 \bar{x}}{n} (n\bar{x})
+ \frac{\bar{x}^{2}}{n} (n) & \text{by (2)} \\[6pt]

\implies & \frac{1}{n} \sum x_{i}^{2}
- 2\bar{x}^{2} + \bar{x}^{2} \\[6pt]

\implies & \frac{1}{n} \sum x_{i}^{2} - \bar{x}^{2}
\end{align}
$$
where $(2)$:
- $\sum x_{i} = n\bar{x}$
- $\sum ^{n}_{i=1} 1 = n$ 

---
## Summary
- Express the [[Moment Generating Function(MGF)|lower-order population moment]] in terms of the [[Statistics Definitions|parameters]]. (because the lower-order are easier to compute)
- Invert the expressions to express the parameter in terms of the [[Moment Generating Function(MGF)|population moment]].
- Replace the [[Moment Generating Function(MGF)|population moments]] using sample moments.

---