# Midterm Notes
The test statistics is 
$$
T(\bar{x}) = \frac{\bar{x} - \mu}{S^{2}/\sqrt{ n }}
$$
To construct a $90\%$ confidence interval, we can use
$$
\left( \frac{(n-1)S^{2}}{\chi^{2}_{0.95, \ df=10}} \ , \quad
\frac{(n-1)S^{2}}{\chi^{2}_{0.05, \ df=10}} \right)
$$

From a given confidence interval like $90\%$ CI is $(42.3, \ 57.7)$, we can 
- first get mean $\bar{X}=\frac{42.3 + 57.7}{2}$
- then, use the formula $\bar{X} \pm t_{0.95, df=10} \times \frac{s}{\sqrt{ n }}$ to get $\frac{s}{\sqrt{ n }}$
- then, we can resubstitute with a diff CI if needed