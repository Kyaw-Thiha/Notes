# MLE-Based CI for $\theta_{0}$
Recall that as per [[MLE Distribution|distribution of MLE]], we have that $\gamma\text{-CI}$ for $\theta_{0}$ is 
$$
\left(\hat\theta - z_{\left(\frac{1+\gamma}{2}\right)}\sqrt{\frac{1}{nI(\theta_0)}}\;,\; \hat\theta + z_{\left(\frac{1+\gamma}{2}\right)}\sqrt{\frac{1}{nI(\theta_0)}}\right)
$$

---
## Estimate of the Fisher Information
When we have unknown parameter $\theta_{0}$, we have two alternatives that give us estimate of the [[Fisher Information|Fisher information]].

### Plug-In Estimate of Fisher Information
$$
nI(\hat\theta) = -E\left[\frac{\partial^2}{\partial\theta^2}\log f(X_1,X_2,\ldots,X_n\mid\theta)\right]\Bigg|_{\theta=\hat\theta}
$$

### Observed Fisher Information
$$
nI(\hat\theta) = -\frac{\partial^2}{\partial\theta^2}\log f(X_1,X_2,\ldots,X_n\mid\theta)\Bigg|_{\theta=\hat\theta}
$$

---
Both of the estimating methods yield estimates. So, using the [[#Plug-In Estimate of Fisher Information|plug-in estimate of Fisher Information]], we get that
$$
\left(\hat\theta - z_{\left(\frac{1+\gamma}{2}\right)}\sqrt{\frac{1}{nI(\hat\theta)}}\;,\; \hat\theta + z_{\left(\frac{1+\gamma}{2}\right)}\sqrt{\frac{1}{nI(\hat\theta)}}\right)
$$

---
## Examples
### CI for $\lambda$ when data follows $Poisson(\lambda)$

Let $X_1,\ldots,X_n \sim Pois(\lambda)$.

**Likelihood**
It has the likelihood function of 
$$
L(\lambda) = \frac{e^{-n\lambda}\,\lambda^{\sum x_i}}{\prod x_i!}
$$

$$
l(\lambda) = -n\lambda + \sum x_i \log\lambda \qquad \text{(dropping the constant } -\log\prod x_i!\text{)}
$$

---
**MLE**
Computing the MLE, we have
$$
l'(\lambda) = -n + \frac{\sum x_i}{\lambda} = 0 \implies \hat\lambda = \bar x
$$
$$
l''(\lambda) = -\frac{\sum x_i}{\lambda^2}
$$

- $\hat\lambda = \bar X$ is the MLE of $\lambda$

---
**Fisher Information**

$$
nI(\lambda) = -E\left[-\frac{\sum X_i}{\lambda^2}\right] = \frac{1}{\lambda^2}\sum E[X_i] = \frac{1}{\lambda^2}\, n\lambda = \frac{n}{\lambda}
$$

- Fisher Information, $nI(\lambda) = \dfrac{n}{\lambda}$

---

**Plug-in Fisher info**

$$
\begin{aligned}
nI(\hat\lambda) &= -E\left[-\frac{\sum X_i}{\lambda^2}\right]\Bigg|_{\lambda=\bar x} \\[6pt]
&= \frac{1}{\lambda^2}\sum E[X_i]\,\Bigg|_{\lambda=\bar x} \\[6pt]
&= \frac{1}{\lambda^2}\, n\lambda\,\Bigg|_{\lambda=\bar x} \\[6pt]
&= \frac{n}{\bar x}
\end{aligned}
$$

- Plug-in estimate, $nI(\hat\lambda) = \dfrac{n}{\bar X}$

---
**$\gamma$-level CI for $\lambda$**

Finally, based on observed data the calculated $\gamma$-CI for $\lambda$ is

$$
\left(\bar X - z_{\left(\frac{1+\gamma}{2}\right)}\sqrt{\frac{\bar X}{n}}\;,\; \bar X + z_{\left(\frac{1+\gamma}{2}\right)}\sqrt{\frac{\bar X}{n}}\right)
$$

i.e. $\bar x \pm z_{\frac{1+\gamma}{2}}\sqrt{\dfrac{\bar x}{n}}$

---
### CI for $\theta$ when data follows $Bern(\theta)$

Let $X_1,\ldots,X_n \sim Bern(\theta)$, so $E[X_i]=\theta$.


**Likelihood**
It has the likelihood function of

$$
L(\theta) = \theta^{\sum x_i}(1-\theta)^{n-\sum x_i}
$$

$$
l(\theta) = \sum x_i \log\theta + \left(n-\sum x_i\right)\log(1-\theta)
$$

---
**MLE**
Computing the MLE, we have

$$
l'(\theta) = \frac{\sum x_i}{\theta} - \frac{n-\sum x_i}{1-\theta} = 0 \implies \hat\theta = \bar x
$$

$$
l''(\theta) = -\frac{\sum x_i}{\theta^2} - \frac{n-\sum x_i}{(1-\theta)^2}
$$

* $\hat\theta = \bar X$ is the MLE of $\theta$

---
**Fisher Information**
$$
\begin{aligned}
nI(\theta) &= -E\left[-\frac{\sum X_i}{\theta^2} - \frac{n-\sum X_i}{(1-\theta)^2}\right] \\[6pt]
&= \frac{1}{\theta^2}\sum E[X_i] + \frac{1}{(1-\theta)^2}\left(n - \sum E[X_i]\right) \\[6pt]
&= \frac{1}{\theta^2}\, n\theta + \frac{1}{(1-\theta)^2}\,(n-n\theta) \\[6pt]
&= \frac{n}{\theta} + \frac{n}{1-\theta} \\[6pt]
&= \frac{n}{\theta(1-\theta)}
\end{aligned}
$$

* Fisher Information, $nI(\theta) = \dfrac{n}{\theta(1-\theta)}$

---
**Plug-in Fisher info**

$$
\begin{aligned}
nI(\hat\theta) &= -E\left[-\frac{\sum X_i}{\theta^2} - \frac{n-\sum X_i}{(1-\theta)^2}\right]\Bigg|_{\theta=\bar x} \\[6pt]
&= \frac{1}{\theta^2}\, n\theta + \frac{1}{(1-\theta)^2}\,(n-n\theta)\,\Bigg|_{\theta=\bar x} \\[6pt]
&= \frac{n}{\theta} + \frac{n}{1-\theta}\,\Bigg|_{\theta=\bar x} \\[6pt]
&= \frac{n}{\theta(1-\theta)}\,\Bigg|_{\theta=\bar x} \\[6pt]
&= \frac{n}{\bar x(1-\bar x)}
\end{aligned}
$$

* Plug-in estimate, $nI(\hat\theta) = \dfrac{n}{\bar X(1-\bar X)}$

---
**$\gamma$-level CI for $\theta$**
Finally, based on observed data the calculated $\gamma$-CI for $\theta$ is

$$
\left(\bar X - z_{\left(\frac{1+\gamma}{2}\right)}\sqrt{\frac{\bar X(1-\bar X)}{n}}\;,\; \bar X + z_{\left(\frac{1+\gamma}{2}\right)}\sqrt{\frac{\bar X(1-\bar X)}{n}}\right)
$$

i.e. $\bar x \pm z_{\frac{1+\gamma}{2}}\sqrt{\dfrac{\bar x(1-\bar x)}{n}}$

---