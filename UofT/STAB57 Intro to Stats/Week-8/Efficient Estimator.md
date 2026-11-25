# Efficient Estimator
Let $T_{1}$ and $T_{2}$ be two different [[Estimator|estimators]] of $\theta$. Efficiency of $T_{1}$ relative to $T_{2}$ is defined as
$$
eff(T_{1}, T_{2}) = \frac{var[T_{2}]}{var[T_{1}]}
$$
where $eff(T_{1}, T_{2}) > 1 \implies T_{1}$ has smaller variance and that $T_{1}$ is more efficient.

> This comparism is meaningful when $T_{1}$ and $T_{2}$ are both unbiased or both have the same bias.

---
## Cramer-Rao Inequality
For any unbiased estimator $T$ of $\theta_{0}$, [[Cramer-Rao Inequality|Cramer-Rao Inequality]] gives a lower bound on its variance $$ var[T] \geq \frac{1}{nI(\theta_{0})} $$ So an unbiased estimator whose variance equals the [[Cramer-Rao Inequality|CRLB]] is the most efficient one possible. Its efficiency relative to the bound is $$ eff(T) = \frac{1/nI(\theta_{0})}{var[T]} \leq 1 $$ where $eff(T)=1$ means $T$ is an efficient estimator. See [[Cramer-Rao Inequality#Examples on Poisson Distribution|the Poisson example]] where $\bar{X}$ attains the CRLB, and [[MLE Distribution|MLE]] which is asymptotically efficient.