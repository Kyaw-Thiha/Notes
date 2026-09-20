# Maximum Likelihood Estimation(MLE)

In [[Maximum Likelihood Estimation(MLE)|MLE]], we pick $\hat{\theta}$ that satisfies $L(\hat{\theta}) \geq L(\theta)$ for all $\theta \in \Omega$.

![image|300](https://notes-media.kthiha.com/Maximum-Likelihood-Estimation(MLE)/3f1d286d32bab1a80488a6ee7b34d854.png)

$\hat{\theta}$ is called the [[Maximum Likelihood Estimation(MLE)|MLE]] of $\theta$.

---
## Intuition
Assume we have two distributions:
- $P_{1}$: $X \sim \text{Unif}\{ 1,2,\dots,10^{3} \}$
- $P_{2}$: $X \sim \text{Unif}\{ 1,2,\dots,10^{6} \}$

If we observe $x=10$,  
$$
\text{Ratio} = \frac{1/10^{3}}{1/10^{6}} = 1000
$$
we have $1000$ times more likely to come from $P_{1}$.

---
## Computation of MLE
### Log-Likelihood
Define the [[Log Likelihood|log-likelihood function]] as
$$
l(\theta) = \ln(L(\theta))
$$
$\ln(x)$ is a $1\text{-}1$ increasing function of $x>0$, so
$$
L(\hat{\theta}) \geq L(\theta)
\text{ for all } \theta\in\Omega \text{ if and only 
if } l(\hat{\theta}) \geq l(\theta)
$$
In other words, if $L(\theta)$ is maximized at $\hat{\theta}$, then $l(\theta)$ will also be maximized at $\hat{\theta}$. Therefore,
$$
l(\theta)
= \ln ( \ \prod^{n}_{i=1} f_{\theta}(x_{i})\  )
= \sum^{n}_{i=1} \ln(f_{\theta}(x_{i}))
$$
We take [[Log Likelihood|log likelihood]] because it is much easier to differentiate a sum than a product.

### Solving
Solve the following equation to obtain $\theta$
$$
\frac{\partial l(\theta)}{\partial \theta }
= 0 
$$

### Checking Maximum
Suppose $\hat{\theta}$ is the solution. We next need to check whether or not
$$
\left.
\frac{\partial^{2} \ l(\theta)}{\partial \theta^{2}}
\right|_{\theta=\hat{\theta}} < 0
$$

---
## Properties of MLE
- [[Maximum Likelihood Estimation(MLE)|MLE]] is not unique.
- [[Maximum Likelihood Estimation(MLE)|MLE]] may not exist.
- The [[Likelihood Function|likelihood function]] may not always be differentiable.

---
### Invariance Property
Let $\hat{\theta}$ be [[Maximum Likelihood Estimation(MLE)|MLE]] of $\theta$ and $\phi(\theta)$ be any $1\text{-}1$ function of $\theta$ defined on $\Omega$.
Then, $\phi(\hat{\theta})$ is the [[Maximum Likelihood Estimation(MLE)|MLE]] of $\phi(\theta)$.

For example, 
- For $\text{Bern}(\theta)$, [[Maximum Likelihood Estimation(MLE)|MLE]] for $\theta$ is $\bar{X}$.
- For $\theta \in [0,1]$, $\theta^{2}$ is a one-one function of $\theta$.
- Hence, the [[Maximum Likelihood Estimation(MLE)|MLE]] for $\theta^{2}$ is $\bar{X}^{2}$.

---
## Examples
### Example-1
**Question**: Let $X_{1}, X_{2}, \dots, X_{n} \overset{iid}{\sim} \text{Poisson}(\lambda)$. Find the [[Maximum Likelihood Estimation(MLE)|MLE]] of $\lambda$.
**Solution**: Recall that the PDF is of
$$
P[X=x] = \frac{e^{-\lambda} \lambda^{x}}{x!}
$$
Computing the [[Likelihood Function|likelihood function]], we get
$$
\begin{align}
L(\lambda)
&= P[X_{1} = x_{1}] * P[X_{2} = x_{2}] *
\dots * P[X_{n} = x_{n}] \\[6pt]
&= \frac{e^{-\lambda} \ \lambda^{x_{1}}} 
{x_{1}!}  
\frac{e^{-\lambda} \ \lambda^{x_{2}}} 
{x_{2}!} \dots  
\frac{e^{-\lambda} \ \lambda^{x_{n}}}{x_{n}!} \\[6pt]
&= \frac{e^{-n\lambda} \ \lambda^{\sum x_{i}}} 
{\prod^{n}_{i=1} x_{i}!}
\end{align}
$$

Taking the [[Log Likelihood|log likelihood]], we get
$$
l(\lambda) = -n\lambda + \sum x_{i} \log \lambda
- \text{const}
$$

Taking the derivative to $0$, we get
$$
\begin{align}
&\frac{d\ l(\lambda)}{d\lambda} = 0 \\[6pt]
\implies & -n + \frac{\sum x_{i}'}{\lambda} + 0
= 0 \\[6pt]
\implies & -n = -\frac{\sum x_{i}}{\lambda} \\[6pt]
\implies & \hat{\lambda} = \frac{\sum x_{i}}{n}  
\\[6pt]
\implies & \hat{\lambda} = \bar{x}
\end{align}
$$

Taking the second-order derivative, we get
$$
\begin{align}
\frac{d^{2} \ l(\lambda)}{d \lambda^{2}}
&= \left. - \frac{\sum x_{i}}{\lambda^{2}}  
\right|_{\lambda = \bar{x}} \\[6pt]
&= - \frac{n\bar{x}}{\bar{x}^{2}} < 0 \\[6pt]
\end{align}
$$
confirming that it is indeed maxima.

---
### Example-2
**Question**: Let $X_{1}, X_{2}, \dots, X_{n} \overset{iid}{\sim} \text{N}(\mu, \sigma^{2}_{0})$ where $\sigma^{2}_{0}$ is known. Find the [[Maximum Likelihood Estimation(MLE)|MLE]] of $\lambda$.

**Solution**: Recall that the PDF is of
$$
f(x) = \frac{1}{\sqrt{ 2x \sigma^{2}_{0} }}
\exp\left[ - \frac{1}{2\sigma^{2}_{0}} (x-\mu)^{2}
\right]
$$
Computing the likelihood function, we get
$$
\begin{align}
L(\mu)
&= f(x_{1}) * f(x_{2}) * \dots * f(x_{n}) \\[6pt]
&= \frac{1}{\sqrt{ 2\pi \sigma^{2} }} \exp \left[ -\frac{1}{2\sigma^{2}} 
(x_{1} - \mu)^{2} \right]  
* \frac{1}{\sqrt{ 2\pi \sigma^{2} }} \exp \left[ -\frac{1}{2\sigma^{2}} 
(x_{2} - \mu)^{2} \right] * \\[6pt]
&  \dots * \frac{1}{\sqrt{ 2\pi \sigma^{2} }} \exp  
\left[ -\frac{1}{2\sigma^{2}} (x_{n} - \mu)^{2} \right] \\[6pt]
&= \left( \frac{1}{2\pi \sigma^{2}} \right)^{n} \exp \left[ -\frac{1} 
{2\sigma^{2}} \sum(x_{i} - \mu)^{2} \right]
\end{align}
$$

Taking the [[Log Likelihood|log likelihood]], we get
$$
l(\mu) = \text{const} - \frac{1}{2\sigma^{2}} \sum(x_{i}-\mu)^{2}
$$
Taking the derivative to $0$, we get
$$
\begin{align}
&\frac{dl(\mu)}{d\mu} = 0 \\[6pt]
\implies &0 - \frac{1}{2\sigma^{2}} \ 2\sum(x_{i} - \mu)(-1) = 0 \\[6pt]
\implies &\sum(x_{i} - \mu) = 0 \\[6pt]
\implies &\sum x_{i} - \sum \mu = 0 \\[6pt]
\implies &\sum x_{i} - n \mu = 0 \\[6pt]
\implies &\hat{\mu} = \frac{\sum x_{i}}{n} = \bar{x}
\end{align}
$$

---
### Example-3
**Question**: Let $X_{1}, X_{2}, \dots, X_{n} \overset{iid}{\sim} \text{Bernoulli}(\theta)$ where $\sigma^{2}_{0}$ is known. Find the [[Maximum Likelihood Estimation(MLE)|MLE]] of $\lambda$.

**Solution**: Recall that the PDF is of
$$
P[X=x] = \theta^{x}(1-\theta)^{(1-x)}
$$

Computing the likelihood function, we get
$$
\begin{align}
L(\theta)
&= P[X_{1} = x_{1}] * P[X_{2} = x_{2}] * \dots * P[X_{n} = x_{n}] \\[6pt]
&= \theta^{x_{1}}(1-\theta)^{1-x_{1}} * \theta^{x_{2}}(1-\theta)^{1-x_{2}} *
\dots * \theta^{x_{n}}(1-\theta)^{1-x_{n}} \\[6pt]
&= \theta^{\sum x_{i}} (1-\theta)^{n - \sum x_{i}}
\end{align}
$$
Taking the [[Log Likelihood|log likelihood]], we get
$$
l(\theta)
= \sum x_{i} \log(\theta) + \left( n - \sum x_{i} \right) \log(1 - \theta)
$$

Taking the derivative to $0$, we get
$$
\begin{align}
&\frac{dl(\theta)}{d\theta} = 0 \\[6pt]
\implies & \frac{\sum x_{i}}{\theta} - \frac{n - \sum x_{i}}{1 - \theta}
= 0 \\[6pt]
\implies & \frac{\sum x_{i}}{\theta} = \frac{n - \sum x_{i}}{1- \theta} \\[6pt]
\implies & \frac{1-\theta}{\theta} = \frac{n - \sum x_{i}}{\sum x_{i}} \\[6pt]
\implies & \frac{1}{\theta} - 1 = \frac{n}{\sum x_{i}} - 1 \\[6pt]
\implies & \frac{1}{\theta} = \frac{n}{\sum x_{i}} \\[6pt]
\implies & \hat{\theta} = \frac{\sum x_{i}}{n} = \bar{x} \\[6pt]
\end{align}
$$

---