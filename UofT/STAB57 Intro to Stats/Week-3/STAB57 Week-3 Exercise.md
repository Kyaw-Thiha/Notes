# Exercises
## Question-1
Assuming $X_{1}, X_{2}, \dots, X_{n} \overset{iid}{\sim} N(\mu, \sigma^{2})$ and using the properties of $\mathcal{X}^{2}$ [[Chi-Squared Distribution|distribution]], calculate the [[Mean Squared Error(MSE)|MSE]] of $S^{2}$ as an [[Estimator|estimator]] of $\sigma^{2}$.

## Solution
### Setup
Let $X_{1}, X_{2}, \dots, X_{n} \overset{iid}{\sim} N(\mu, \sigma^{2})$ and let the [[Sample Variance|sample variance]] be
$$
S^{2} = \frac{1}{n-1} \sum ^{n}_{i=1} (X_{i} - \bar{X})^{2}
$$
We want to find [[Mean Squared Error(MSE)|MSE]] of $S^{2}$ as an [[Estimator|estimator]] of $\sigma^{2}$:
$$
\text{MSE}(S^{2}) = \mathbb{E}[(S^{2} - \sigma^{2})^{2}]
$$

---
### Use Chi-Square Property
For a normal sample, 
$$
\frac{(n-1)S^{2}}{\sigma^{2}} \sim \mathcal{X}^{2}_{n-1}
$$
Let $Y = \frac{(n-1)S^{2}}{\sigma^{2}}$. Then, $Y \sim \mathcal{X}^{2}_{n-1}$ so we get
$$
S^{2} = \frac{\sigma^{2}}{n-1} Y
$$
For a [[Chi-Squared Distribution|chi-square random variable]] with $k$ degrees of freedom, we have
$$
E(Y) = k \quad \ \quad Var(Y) = 2k
$$
Taking $k=n-1$, we have
$$
E(S^{2}) 
= \frac{\sigma^{2}}{n-1} E(y)
= \frac{\sigma^{2}}{n-1} (n-1)
= \sigma^{2}
$$
Since $E(S^{2}) = \sigma^{2}$, $S^{2}$ is unbiased for $\sigma^{2}$ and its bias is zero.

---
### Variance of $S^{2}$
Using the variance of [[Chi-Squared Distribution|chi-square random variable]], 
$$
Var(S^{2}) = \left( \frac{\sigma^{2}}{n-1} \right)^{2} \ Var(Y)
$$
Since $Var(Y) = 2(n-1)$, 
$$
\begin{align}
Var(S^{2}) &= \frac{\sigma^{4}}{(n-1)^{2}} 2(n-1) \\[6pt]
&= \frac{2\sigma^{4}}{n-1}
\end{align}
$$

---
### Calculate the MSE
Recall from [[Mean Squared Error(MSE)|MSE]] that 
$$
\text{MSE}(\hat{\theta}) = Var(\hat{\theta}) + [\text{Bias}(\hat{\theta})]^{2}
$$
Since the bias of $S^{2}$ is zero,
$$
\boxed{ \ \text{MSE}(S^{2}) = \frac{2\sigma^{4}}{n-1} \ }
$$

---
# Question-2

Assuming $X_1,X_2,\dots,X_n\overset{iid}{\sim}N(\mu,\sigma^2)$ and using the properties of $\mathcal{X}^2$ distribution, calculate the MSE of $\hat{\sigma}^2$ as an estimator of $\sigma^2$, where

$$
\hat{\sigma}^2=\frac{n-1}{n}S^2
$$

## Solution

### Setup

Let $X_1,X_2,\dots,X_n\overset{iid}{\sim}N(\mu,\sigma^2)$ and let

$$
\hat{\sigma}^2=\frac{n-1}{n}S^2
$$

We want to find MSE of $\hat{\sigma}^2$ as an estimator of $\sigma^2$:

$$
\text{MSE}(\hat{\sigma}^2)
=\mathbb{E}[(\hat{\sigma}^2-\sigma^2)^2]
$$

---

### Use Chi-Square Property

For a normal sample,

$$
\frac{(n-1)S^2}{\sigma^2}\sim\mathcal{X}^2_{n-1}
$$

Let $Y$ be
$$
Y=\frac{(n-1)S^2}{\sigma^2}.
$$

Then, $Y\sim\mathcal{X}^2_{n-1}$ so we get

$$
S^2=\frac{\sigma^2}{n-1}Y
$$

Since

$$
\hat{\sigma}^2=\frac{n-1}{n}S^2,
$$

we have

$$
\hat{\sigma}^2
=\frac{n-1}{n}\frac{\sigma^2}{n-1}Y
=\frac{\sigma^2}{n}Y
$$

For a chi-square random variable with $k$ degrees of freedom, we have

$$
E(Y)=k
\qquad\qquad
Var(Y)=2k
$$

Taking $k=n-1$, we have

$$
\begin{align}
E(\hat{\sigma}^2)
&=\frac{\sigma^2}{n}E(Y)\\[6pt]
&=\frac{\sigma^2}{n}(n-1)\\[6pt]
&=\frac{n-1}{n}\sigma^2
\end{align}
$$

Since

$$
E(\hat{\sigma}^2)\neq\sigma^2,
$$

$\hat{\sigma}^2$ is biased for $\sigma^2$.

Therefore,

$$
\begin{align}
\text{Bias}(\hat{\sigma}^2)
&=E(\hat{\sigma}^2)-\sigma^2\\[6pt]
&=\frac{n-1}{n}\sigma^2-\sigma^2\\[6pt]
&=-\frac{\sigma^2}{n}
\end{align}
$$

---

### Variance of $\hat{\sigma}^2$

Using the variance of chi-square random variable,

$$
Var(\hat{\sigma}^2)
=\left(\frac{\sigma^2}{n}\right)^2 Var(Y)
$$

Since $Var(Y)=2(n-1)$,

$$
\begin{align}
Var(\hat{\sigma}^2)
&=\frac{\sigma^4}{n^2}2(n-1)\\[6pt]
&=\frac{2(n-1)\sigma^4}{n^2}
\end{align}
$$

---

### Calculate the MSE

Recall from MSE that

$$
\text{MSE}(\hat{\theta})
=Var(\hat{\theta})+[\text{Bias}(\hat{\theta})]^2
$$

Therefore,

$$
\begin{align}
\text{MSE}(\hat{\sigma}^2)
&=\frac{2(n-1)\sigma^4}{n^2}
+\left(-\frac{\sigma^2}{n}\right)^2\\[6pt]
&=\frac{2(n-1)\sigma^4}{n^2}
+\frac{\sigma^4}{n^2}\\[6pt]
&=\frac{(2n-1)\sigma^4}{n^2}
\end{align}
$$

Thus,

$$
\boxed{\text{MSE}(\hat{\sigma}^2)
=\frac{(2n-1)\sigma^4}{n^2}}
$$
---
