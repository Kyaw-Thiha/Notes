# Test of Hypothesis using Confidence Interval
Similar to constructing $\gamma$-[[Confidence Interval|confidence intervals]], 
- we keep the $\gamma$ part of the interval
- instead of discarding the corners $(1-\gamma)$, we define them as rejection regions.

Constructing a $\gamma$-level [[Confidence Interval|confidence interval]] for $\mu$ and checking whether $\mu_{0}$ is inside or not is equivalent to testing the [[Hypothesis Test|hypothesis]] of $\mu=\mu_{0}$ at $(1-\gamma)$ level of significance.

---
## Hypothesis Testing
We fail to reject $H_{0}$ if
$$
\begin{align}
z_{\frac{\alpha}{2}} &< \frac{\bar{X} - \mu}{\sigma/\sqrt{ n }}  
< z_{1-\frac{\alpha}{2}} \\[6pt]
\bar{X} + z_{\frac{\alpha}{2}} \frac{\sigma}{\sqrt{ n }}
&< \mu < \bar{X} + z_{1 - \frac{\alpha}{2}} \frac{\sigma}{\sqrt{ n }} \\[6pt]
\end{align}
$$
If $\sigma = 1 - \alpha$ or $\alpha = 1 - \sigma$, we get
$$
\bar{X} + z_{\frac{1-\alpha}{2}} \frac{\sigma}{\sqrt{ n }}
< \mu < \bar{X} + z_{\frac{1+\alpha}{2}} \frac{\sigma}{\sqrt{ n }}
$$
which is the $\alpha\text{-level}$ [[Confidence Interval|confidence interval]] for $\mu$.

---

