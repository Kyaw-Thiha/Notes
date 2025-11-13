# Gaussian CCM

`Gaussian CCM` is a [[Class-Conditional Model]] that assume likelihoods to be [[Gaussian Distribution]]
$$
p(x|c_{i}) = \frac{1}{(2\pi)^D |\Sigma_{i}|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu_{i}) \Sigma^{-1}_{i} (x - \mu_{i}) \right)
$$

This allows us to simplify the calculation.

---
## Derivation

Continuing from [[Binary Classification CCM]], assume each class have the same prior to simplify the algebra.
$$
\begin{align}
a(x) &= \log\left( \frac{p(x|c_{1})}{p(x|c_{2})} \times \frac{p(c_{1})}{p(c_{2})} \right) \\[6pt]
&= \log\left( \frac{p(x|c_{1})}{p(x|c_{2})} \right) \\[6pt]
&= \log p(x|c_{1}) - \log p(x|c_{2})
\end{align}
$$

Since the likelihoods are `Gaussian`, we can derive that
$$
\log p (x|c_{i}) 
= \frac{1}{2} \log((2\pi)^D |\Sigma_{i}|) - \frac{1}{2}(x - \mu_{i})^T \Sigma^{-1}_{i} (x - \mu_{i})
$$

Using it, we can further simplify $a(x)$ as
$$
\begin{align}
a(x)  
&= \log p(x | c_{1}) - \log p(x|c_{2}) \\[6pt] 

&= -\frac{1}{2} \log((2\pi)^D |\Sigma_{1}|)
-\frac{1}{2} (x - \mu_{1})^T \Sigma^{-1}_{1} (x - \mu_{1})
+ \frac{1}{2} \log((2\pi)^D |\Sigma_{2}|)
-\frac{1}{2} (x - \mu_{2})^T \Sigma^{-1}_{2} (x - \mu_{2}) \\[6pt]

&= \underbrace{-\frac{1}{2} \log|\Sigma_{1}| + \frac{1}{2} \log |\Sigma_{2}|}_{\text{constant term}}
\underbrace{- \frac{1}{2} (x - \mu_{1})^T \Sigma_{1}^{-1} (x  - \mu_{1})
- \frac{1}{2} (x - \mu_{2})^T \Sigma_{2}^{-1} (x  - \mu_{2})}_{\text{quadratic term}}
\end{align}
$$

`Same CoVariance Assumption`
To get a good analytical solution, consider the situation where the covariances of the 2 distributions are the same.
Hence, $\Sigma_{1} = \Sigma_{2} = \Sigma$
$$
\begin{align}
a(x)  
&= -\frac{1}{2} \log|\Sigma_{1}| + \frac{1}{2} \log |\Sigma_{2}|
- \frac{1}{2} (x - \mu_{1})^T \Sigma_{1}^{-1} (x  - \mu_{1})
- \frac{1}{2} (x - \mu_{2})^T \Sigma_{2}^{-1} (x  - \mu_{2}) \\[6pt]

&= -\frac{1}{2} (x^T \Sigma^{-1} x - 2x^T\Sigma^{-1}\mu_{1} + \mu_{1}^T \Sigma^{-1}\mu_{1})
+\frac{1}{2} (x^T \Sigma^{-1} x - 2x^T\Sigma^{-1}\mu_{2} + \mu_{2}^T \Sigma^{-1}\mu_{2}) \\[6pt]

&= x^T \Sigma^{-1} \mu_{1} - x^T\Sigma^{-1}\mu_{2}
\underbrace{- \frac{1}{2} \mu_{1}^T\Sigma^{-1}\mu_{1} + \frac{1}{2} \mu_{2}^T \Sigma^{-1} \mu_{2}}
_{\text{Constant term } b} \\[6pt]

&= \underbrace{x^T \Sigma^{-1} (\mu_{1} - \mu_{2}) + b}_{\text{Linear function of } x}
\end{align}
$$

---
## Remarks
- If the likelihoods $p(x|c_{i})$ are not [[Gaussian Distribution]], then `Gaussian CCM` perform horribly.
- No. of parameters in `Covariance Matrix`: $\frac{D(D+1)}{2} \sim O(D^2)$ 
  Hence, no. of parameters is very large in high dimensional data
- We need a huge amount of data to estimate the parameters

In order to reduce the number of parameters, we can use [[Naive Bayes]], where we assume `covariance matrix` is diagonal.

---
## See Also
- [[Class-Conditional Model]]
- [[Binary Classification CCM]]
- [[Naive Bayes]]