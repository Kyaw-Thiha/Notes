# Week-9 Tutorial
$$
f(x) = \frac{\lambda^{3}}{2} X^{2} e^{-\lambda X} \quad, \quad X>0
$$

## Question-(a)
Find MLE of $\lambda$

### Answer
$$
\begin{align}
L(\lambda)  
&= \prod ^{n}_{i=1}\left( \frac{\lambda^{2}}{2} X_{i}^{2} e^{-\lambda X_{i}} \right) \\[6pt]
&= \frac{\lambda^{3n}}{2^{n}} \left( \prod ^{n}_{i=1 X_{i}^{2}} e^{-\lambda \sum ^{n}_{i=1} X_{i}} \right)
\end{align}
$$

Taking log likelihood, we get
$$
L(\lambda) 
= 3n \log \lambda - n \log 2 + 2 \sum ^{n}_{i=1} \log X_{i}
- \lambda \sum ^{n}_{i=1} X_{i}
$$

Taking the derivative, we get
$$
L'(\lambda) = \frac{3n}{\lambda} - \sum ^{n}_{i=1} X_{i}
$$
Taking the first derivative to $0$
$$
\begin{align}
\frac{3n}{\lambda} - \sum ^{n}_{i=1} X_{i} &= 0 \\[6pt]
\lambda &= \frac{3n}{\sum ^{n}_{i=1} X_{i}}
\end{align}
$$
Taking the second derivative, we get
$$
L''(\lambda) = -\frac{3n}{\lambda^{2}} < 0
$$

---
## Question-(b)
Find the Fisher Information for the entire symbol.
Review: $In(\theta) = -E_{\theta}[L''(\theta)]$

### Solution
$$
\begin{align}
In(\lambda) &= -E_{\lambda} \left[ -\frac{3n}{\lambda^{2}} \right]
\\[6pt]
In(X) &= - \left( -\frac{3n}{\lambda^{2}} \right) = \frac{3n}{\lambda^{2}}
\end{align}
$$

---
## Question-(c)
Construct the plug-in confidence interval.
Review: 

For a large sample, the MLE is approximately normal with
$$
SE(\hat{\theta}) = \sqrt{ \frac{1}{In(\hat{\theta})} } \qquad , \qquad 
\text{variance} = \frac{1}{In(\theta)}
$$

Also $\alpha = 1 - \gamma$ (Middle area is $\gamma$)

Interval = $\hat{\theta} \pm Z_{1-\frac{\alpha}{2}} SE(\hat{\theta})$

---
### Solution
$$
\hat{SE(\hat{\theta})} = \sqrt{ \frac{1}{\frac{3n}{\hat{\lambda}^{2}}} }
= \sqrt{ \frac{\hat{\lambda}^{2}}{3n} }
= \frac{|\hat{\lambda}|}{\sqrt{ 3n }}
= \frac{\hat{\lambda}}{\sqrt{ 3n }}
$$
where $\hat{\lambda} = \frac{3}{\bar{X}} > 0$.

$$
\text{Interval}: \left[ 
\frac{3}{\bar{X}} \left( 1 - \frac{Z_{\frac{1+\gamma}{2}}}{\sqrt{ 3n }} \right)  , \quad
\frac{3}{\bar{X}} \left( 1 + \frac{Z_{\frac{1+\gamma}{2}}}{\sqrt{ 3n }} \right) 
\right]
$$

---
## Question-(d)
Give CRLB.

$$
var_{\theta}(T) \geq \frac{\left[ \frac{d}{d\theta} E_{\theta}[T] \right]^{2}}{In(\theta)}
$$
For unbiased estimate $\theta$,
$$
var_{\theta}(T) \geq \frac{1}{In(\theta)}
$$

---
### Solution
$$
CRLB 
= \frac{1}{In(X)}
= \frac{1}{\frac{3n}{\lambda^{2}}}
= \frac{\lambda^{2}}{3n}
$$

---