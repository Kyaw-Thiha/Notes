# Sufficient Statistics
> A [[Test Statistics|statistics]] $T(X_{1}, X_{2}, \dots, X_{n})$ is said to be [[Sufficient Statistics|sufficient]] for $\theta$ if the conditional distribution of $X_{1}, X_{2}, \dots, X_{n}$, given $T=t$, does not depend on $\theta$.

In other words, once we have the value of the [[Sufficient Statistics|sufficient statistics]], the actual sample observation don't add any more information about the parameter.

---
## Example
Suppose $X_1, X_2, X_3 \overset{iid}{\sim} \text{Poisson}(\lambda)$. Verify that $T = \sum_{i=1}^{3} X_i$ is a sufficient statistic for $\lambda$.

### Solution
$$
\begin{aligned}
&P\left[X_1 = x_1,\, X_2 = x_2,\, X_3 = x_3 \,\middle|\, \sum X_i = t\right] \\[6pt]
&= \frac{P\left[X_1 = x_1,\, X_2 = x_2,\, X_3 = x_3 \;\&\; \sum X_i = t\right]}{P\left[\sum X_i = t\right]} \\[2ex]
&= \frac{P\left[X_1 = x_1,\, X_2 = x_2,\, X_3 = t - x_1 - x_2\right]}{P\left[\sum X_i = t\right]} \\[2ex]
&= \frac{P[X_1 = x_1] \cdot P[X_2 = x_2] \cdot P[X_3 = t - x_1 - x_2]}{P\left[\sum X_i = t\right]} \\[2ex]
&= \frac{\dfrac{e^{-\lambda}\lambda^{x_1}}{x_1!} \cdot \dfrac{e^{-\lambda}\lambda^{x_2}}{x_2!} \cdot \dfrac{e^{-\lambda}\lambda^{t-x_1-x_2}}{(t-x_1-x_2)!}}{\dfrac{e^{-3\lambda}(3\lambda)^t}{t!}} \\[2ex]
&= \text{expression free of } \lambda.
\end{aligned}
$$

---
## Factorization Theorem
$T(X_{1}, X_{2}, \dots, X_{n})$ is said to be [[Sufficient Statistics|sufficient]] for $\theta$ if thhe joint probability function factors in the form 
$$
f(x_1, x_2, \ldots, x_n \mid \theta) = g\left[T(x_1, x_2, \ldots, x_n), \theta\right] \ast h(x_1, x_2, \ldots, x_n)
$$
where
- $h(x_1, x_2, \ldots, x_n)$ is a function of the sample observations only
- $g\left[T(x_1, x_2, \ldots, x_n), \theta\right]$ involves $\theta$ and the sufficient statistic $T$

---
## Examples
### Example-1
Let $X_{1}, X_{2}, \dots, X_{n} \sim Pois(\lambda)$. Then, 
$$
\begin{aligned}
L(\lambda) &= \frac{e^{-n\lambda}\,\lambda^{\sum x_i}}{\prod x_i!} \\[2ex]
&= \underbrace{\bbox[#CFE2F3,5px]{\color{black}e^{-n\lambda}\,\lambda^{\sum x_i}}}_{g(T,\lambda)} \cdot \underbrace{\bbox[#D9EAD3,5px]{\color{black}\frac{1}{\prod x_i!}}}_{h(x_1,\ldots,x_n)}
\end{aligned}
$$

By the [[#Factorization Theorem|factorization theorem]], $\sum x_{i}$ is sufficient for $\lambda$.

---
### Example-2
Let $X_{1}, X_{2}, \dots, X_{n} \sim \mathcal{N}(\lambda)$. Then, 
$$
\begin{aligned}
L(\mu, \sigma^2) &= \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^n \exp\left[-\frac{1}{2\sigma^2}\sum (x_i - \mu)^2\right] \\[2ex]
&= \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^n \exp\left[-\frac{1}{2\sigma^2}\left(\sum x_i^2 - 2\mu \sum x_i + n\mu^2\right)\right] \\[2ex]
&= \left(\frac{1}{\sqrt{\sigma^2}}\right)^n \exp\left[-\frac{1}{2\sigma^2}\left(\sum x_i^2 - 2\mu \sum x_i + n\mu^2\right)\right] \ast \left(\frac{1}{\sqrt{2\pi}}\right)^n
\end{aligned}
$$
Hence, $\sum x_{i}$ $\&$ $\sum x_{i}^{2}$ are jointly sufficient for $\mu$ and $\sigma^{2}$.

---
### Example-3
Let $X_{1}, X_{2}, \dots, X_{n} \sim \exp(\lambda)$. Then, 
$$
f(x) = \lambda e^{-\lambda x}
$$
Hence, we have
$$
L(\lambda) = \lambda^{n} e^{-\lambda \sum x_{i}} \cdot 1
$$
Therefore, we get that $\sum x_{i}$ is [[Sufficient Statistics|sufficient statistics]] for $\lambda$.

---
## Link to MLE
Note that when we maximize [[Likelihood Function|likelihood]], we maximize $g\left[T(x_1, x_2, \ldots, x_n), \theta\right]$. 
$$
\begin{aligned}
&L(\theta) = g\left[T, \theta\right] \ast h(x_1, \ldots, x_n) \\[2ex]
\implies \quad &\ell(\theta) = \log g\left[T, \theta\right] + \log h(x_1, \ldots, x_n) \\[2ex]
\implies \quad &\frac{d}{d\theta}\,\ell(\theta) = \frac{d}{d\theta}\log g\left[T, \theta\right] + 0 \\[2ex]
\implies \quad &\hat{\theta} = \text{func. of } T
\end{aligned}
$$
[[Maximum Likelihood Estimation(MLE)|MLE]] is a function of $T$.

---

