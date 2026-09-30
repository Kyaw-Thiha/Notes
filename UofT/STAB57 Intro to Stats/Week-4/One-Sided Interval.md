# One-Sided Interval
A [[One-Sided Interval|one-sided interval]] looks like
$$
P[-\infty \leq \psi(\theta) \leq u(X_{1}, X_{2}, \dots, X_{n})] \geq \gamma
$$
or
$$
P[l(X_{1}, X_{2}, \dots, X_{n}) \leq \psi(\theta) \leq \infty] \geq \gamma
$$

---
## Example
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

4. Using t-table or $R[qt(0.95, df=9)]$, we get
$$
t_{0.95(9)} \approx 1.833
$$
5. Hence the 0.95-CI for $\mu$ is
$$
\left( -\infty, \ 4.88 + 1.833 * \frac{0.696}{\sqrt{ 10 }} \right)
$$

---

