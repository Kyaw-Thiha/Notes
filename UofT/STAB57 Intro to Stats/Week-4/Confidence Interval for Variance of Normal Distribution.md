# Confidence Interval for Variance of Normal Distribution
Recall that
$$
\frac{(n-1)S^{2}}{\sigma^{2}} \sim \mathcal{X}^{2}_{(n-1)}S
$$
We can then use it to derive
$$
\begin{align}
&P\left[ k_{1} \leq \frac{(n-1)S^{2}}{\sigma^{2}} \leq k_{2} \right] = \gamma
\\[6pt]
\implies  &P\left[ \frac{k_{1}}{(n-1)S^{2}} \leq \frac{1}{\sigma^{2}}  
\leq \frac{k_{2}}{(n-1)S^{2}} \right] = \gamma  \\[6pt]
\implies &P\left[ \frac{(n-1)S^{2}}{k_{2}} \leq \sigma^{2} \leq  
\frac{(n-1)S^{2}}{k_{1}} \right] = \gamma
\end{align}
$$
where
- $k_{1} = \mathcal{X}^{2}_{\frac{1-\gamma}{2}, df=n-1}$
- $k_{2} = \mathcal{X}^{2}_{\frac{1+\gamma}{2}, df=n-1}$

---
## Example
**Question**: Let $(4.7, 5.5, 4.4, 3.3, 4.6, 5.3, 5.2, 4.8, 5.7, 5.3) \overset{iid}{\sim} \mathbb{N}(\mu, \sigma_{0}^{2})$ with $\sigma^{2}_{0}=0.5$.
Calculate the $0.95$-confidence interval for $\mu$.

**Solution**:
1. We have $n=10$, 
2. and a mean of
$$
\bar{X} = \frac{1}{10}(4.7 + 5.5 + \dots + 5.3) = 4.88
$$
3. 

$$
(n-1)s^{2}
= \sum(x_{i} - \bar{x})^{2} 
= \sum x_{i}^{2} - n(\bar{x})^{2}
= 4.356
$$
4. Since $\gamma=0.95$, we have 
$$\frac{1+\gamma}{2} = 0.975$$
5. Using $\mathcal{X}^{2}$-table or $R[qchisq(0.025, df=9)]$ and $qchisq(0.975, df=9)$, we have
$$
\chi^{2}_{0.025(9)} \approx 2.7 
\ \text{ and } \ 
\chi^{2}_{0.975(9)} \approx 19.023
$$
6. Hence the 0.95-CI for $\mu$ is
$$
\left( \frac{4.356}{19.023}, \ \frac{4.356}{2.7} \right)
= (0.229, \ 1.613)
$$

---
## Notes
- $\mathcal{X}^{2}$ is not a symmetric distribution.
- Its shape depends on its degrees of freedom.
- Using $\chi^{2}_{\frac{1-\gamma}{2}(n-1)}$ and $\chi^{2}_{\frac{1+\gamma}{2}(n-1)}$ as two ends may not result in the shortest length.

---
