# Distribution of MLE
Under [[#Conditions|some conditions]], the [[MLE Distribution|distribution of MLE]] as $n\to \infty$ is
$$
\frac{(\hat{\theta} - \theta_{0})}{\sqrt{ 1/nI(\theta_{0}) }}
\overset{D}{\to} N(0, 1)
$$
Through this claim, it would follow that for large $n$,
$$
E[\hat{\theta}] = \theta_{0}
\qquad \qquad
V[\hat{\theta}] = \frac{1}{nI(\theta_{0})}
$$
And we have a sampling distribution through asymptotic nature of [[Maximum Likelihood Estimation(MLE)|MLE]].

---
### Conditions
We have the following conditions:
- The data are iid from $f(x\mid\theta_0)$, and $\theta$ is identifiable.
- The support of $f$ doesn’t depend on $\theta$, and $\theta_0$ is an interior point of the parameter space.
- $\log f$ is smooth enough in $\theta$, and you can swap $\partial/\partial\theta$ and $\int$ (the same step as in proof).
- $0<I(\theta_0)<\infty$
- $\hat\theta$ is consistent.

---
## Proof
### Preliminary
Given how $\hat{\theta}$ is defined, we have $l'(\hat{\theta}) = 0$.
And as proven before, we have $E[l'(\theta_{0})] = 0$.

---
### Asymptotic distribution of the score function
**Claim.** We have that 
$$
\boxed{
l'(\theta_0) \xrightarrow{D} N\big(0,\, nI(\theta_0)\big) \implies \frac{1}{n}\, l'(\theta_0) \xrightarrow{D} N\left(0,\, \frac{1}{n}I(\theta_0)\right) }
$$
**Proof.** This can be proven as 
$$
\begin{aligned}
l'(\theta_0) &\to N\big(0,\, nI(\theta_0)\big) \\[6pt]
\frac{1}{n}\, l'(\theta_0) &\to N\left(0,\, \frac{1}{n^2}\, nI(\theta_0)\right) 
\\[6pt]
\frac{1}{n}\, l'(\theta_0) &\xrightarrow{D} N\left(0,\, \frac{I(\theta_0)}{n}\right)
\end{aligned}
$$

---
### Convergence of Average Second Derivative
**Claim.** By [[Law of Large Numbers(LLN)|laws of large numbers]], we have that
$$
\frac{1}{n}\, l''(\theta_0) \xrightarrow{P} -I(\theta_0)
$$
**Proof.** Since $l(\theta)=\sum_{i=1}^n \log f(X_i\mid\theta)$, $$ \begin{aligned} \frac{1}{n}\, l''(\theta_0) &= \frac{1}{n}\sum_{i=1}^{n} \frac{\partial^2}{\partial\theta^2}\log f(X_i\mid\theta)\Big|_{\theta=\theta_0} && \text{(definition of } l'' \text{)} \\[8pt] &\xrightarrow{P}\; E\left[\frac{\partial^2}{\partial\theta^2}\log f(X\mid\theta)\Big|_{\theta=\theta_0}\right] && \text{(LLN: sample mean of iid terms)} \\[8pt] &= -I(\theta_0) && \text{(information equality)} \end{aligned} $$ Multiplying by $-1$ ([[Continuous Mapping Theorem|continuous mapping theorem]]), $$ \boxed{-\frac{1}{n}\, l''(\theta_0) \xrightarrow{P} I(\theta_0)} \qquad \blacksquare $$

---
### Applying Taylor series on $l'(\hat\theta)$

Expanding $f(x)$ around $a$:

$$
f(x) = f(a) + (x-a)f'(a) + (x-a)^2\frac{f''(a)}{2!} + \cdots
$$

We have,

$$
f(x) = l'(\hat\theta), \qquad a = \theta_0
$$

$$
\begin{aligned}
&l'(\hat\theta) = l'(\theta_0) + (\hat\theta-\theta_0)\,l''(\theta_0) + \text{higher order terms} && \text{(neglect)} \\[6pt]
\implies &l'(\hat\theta) \approx l'(\theta_0) + (\hat\theta-\theta_0)\,l''(\theta_0) \\[6pt]
\implies &0 \approx l'(\theta_0) + (\hat\theta-\theta_0)\,l''(\theta_0) \\[6pt]
\implies &\hat\theta-\theta_0 \approx -\frac{l'(\theta_0)}{l''(\theta_0)} = \frac{(1/n)\,l'(\theta_0)}{-(1/n)\,l''(\theta_0)}
\end{aligned}
$$

- numerator $\xrightarrow{D} N\left(0,\, \dfrac{1}{n}I(\theta_0)\right)$
- denominator $\xrightarrow{P} I(\theta_0)$

Let $Y = \dfrac{1}{n}\,l'(\theta_0)$. Then

$$
\begin{aligned}
Y &\to N\left(0,\, \frac{I(\theta_0)}{n}\right) \\[6pt]
\frac{Y}{I(\theta_0)} &\to N\left(0,\, \frac{1}{\{I(\theta_0)\}^2}\cdot\frac{I(\theta_0)}{n}\right) \\[6pt]
&\to N\left(0,\, \frac{1}{nI(\theta_0)}\right)
\end{aligned}
$$

This gives us

$$
\hat\theta-\theta_0 \xrightarrow{D} N\left(0,\, \frac{1}{nI(\theta_0)}\right)
\implies
\boxed{\frac{\hat\theta-\theta_0}{\sqrt{1/nI(\theta_0)}} \xrightarrow{D} N(0,1)}
$$

As $n\to\infty$,

$$
\hat\theta \to N\left(\theta_0,\, \frac{1}{nI(\theta_0)}\right)
$$

$$
\boxed{E[\hat\theta] = \theta_0, \quad \hat\theta \text{ is unbiased}} \qquad \text{as } n\to\infty
$$

**MLE is asymptotically unbiased.**

---
**Summary**
- $l'(\theta_0) \to N\big(0,\, nI(\theta_0)\big)$
- $\hat\theta \to N\left(\theta_0,\, \dfrac{1}{nI(\theta_0)}\right)$

---
## Claims about MLE
- [[Maximum Likelihood Estimation(MLE)|MLE]] is asymptotically unbiased.
- [[Maximum Likelihood Estimation(MLE)|MLE]] is function of [[Sufficient Statistics|sufficient statistics]].
- [[Maximum Likelihood Estimation(MLE)|MLE]] is [[Consistent Estimator|consistent]].
- [[Maximum Likelihood Estimation(MLE)|MLE]] is asymptotically efficient.
- And we have

$$
\frac{(\hat\theta-\theta_0)}{\sqrt{1/nI(\theta_0)}} \xrightarrow{D} N(0,1)
$$

$$
\hat\theta \to N\left(\theta_0,\, \frac{1}{nI(\theta_0)}\right)
$$

---
