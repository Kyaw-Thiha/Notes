# Fisher Information
The [[Fisher Information|Fisher Information]] is defined as 
$$
I(\theta_0) = \text{var}\left[S(\theta \mid X)\Big|_{\theta=\theta_0}\right] = E\left[\left(\frac{\partial}{\partial\theta}\log f(X \mid \theta)\Big|_{\theta=\theta_0}\right)^2\right]
$$
which is essentially the amount of information that each observable random variable $X$ contains about $\theta$.

---
## Poisson Distribution
For Poisson distribution, we have that
$$
I(\lambda_0) = \text{var}\left[S(\lambda \mid X)\Big|_{\lambda=\lambda_0}\right] = \text{var}\left[-1 + \frac{X}{\lambda_0}\right] = \frac{1}{\lambda_0^2}\,\text{var}[X] = \frac{1}{\lambda_0}
$$

---
## Fisher Information from a Sample
We can get [[Fisher Information|Fisher Information]] from a set of sample of size $n$ as
$$
\begin{aligned}
\text{var}\left[S(\theta \mid X_1, X_2, \ldots, X_n)\Big|_{\theta=\theta_0}\right]
&= \text{var}\left[\sum_i S(\theta \mid X_i)\Big|_{\theta=\theta_0}\right] \\[2ex]
&= \sum_i \text{var}\left[S(\theta \mid X_i)\Big|_{\theta=\theta_0}\right] \\[2ex]
&= nI(\theta_0)
\end{aligned}
$$

---
## Easier Way of Calculating $nI(\theta_{0})$
It can be shown that 
$$
I(\theta_0) = E\left[\left(\frac{\partial}{\partial\theta}\log f(X \mid \theta)\Big|_{\theta=\theta_0}\right)^2\right] = \bbox[#CFE2F3,5px]{\color{black}-E\left[\frac{\partial^2}{\partial\theta^2}\log f(X \mid \theta)\Big|_{\theta=\theta_0}\right]}
$$
Often in practice, we use the second derivative of the [[Log Likelihood|log likelihood]] to calculate $nI(\theta_{0})$.

---
### Steps
The steps to calculate can be summarized as 
- Write down the log-likelihood in terms of random variable $X$
- Differentiate twice with respect to $\theta$ and then put $\theta = \theta_0$
- Take Expectation over $X$ and finally multiply by $(-1)$

---
### Example
For $X_1, X_2, \ldots, X_n \overset{iid}{\sim} \text{Pois}(\lambda)$, we have
$$
nI(\lambda_0) = -E\left[\frac{\partial^2}{\partial\lambda^2}\left(-n\lambda + \sum_i X_i \log\lambda\right)\Big|_{\lambda=\lambda_0}\right] = -E\left[-\frac{\sum_i X_i}{\lambda_0^2}\right] = \frac{n}{\lambda_0}
$$

---

