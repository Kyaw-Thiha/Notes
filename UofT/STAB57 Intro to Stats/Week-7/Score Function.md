# Score Function
The [[Score Function|score function]] $S(\theta)$ is the derivative of the [[Log Likelihood|log likelihood]].
$$
S(\theta) 
= \frac{\delta l(\theta)}{\delta \theta}
= l'(\theta)
$$

---
## Score Equation
We can take $S(\theta) = 0$ and the solution to this equation is the [[Maximum Likelihood Estimation(MLE)|MLE]].
We can thus, say that
$$
S(\theta) \mid_{\theta = \hat{\theta}} \ = \  0
$$

---
## Score as a Random Variable
For the random variable $X_{i},$ 
$$
S(\theta \mid X_{i}) = \frac{\partial}{\partial \theta} \log f(X_{i} \mid \theta)
$$

---
### IID Random Variables
For an $\text{iid}$ of $(X_{1}, X_{2}, \dots, X_{n})$, we have
$$
L(\theta) = \prod_{i=1}^{n} f(X_i \mid \theta) 
$$
Taking the [[Log Likelihood|log likelihood]], we have
$$
\ell(\theta) = \sum_{i=1}^{n} \log f(X_i \mid \theta) 
$$
Finding the derivative, we get
$$
\begin{aligned}
S(\theta \mid X_1, \ldots, X_n) = \frac{d}{d\theta}\,\ell(\theta) &= \frac{\partial}{\partial\theta}\sum_{i} \log f(X_i \mid \theta) \\[2ex]
&= \bbox[#CFE2F3,5px]{\color{black}\sum_{i} \frac{\partial}{\partial\theta}\log f(X_i \mid \theta)} \\[2ex]
&= \bbox[#D9EAD3,5px]{\color{black}\sum_{i} S(\theta \mid X_i)}
\end{aligned}
$$
---
## Randomness of Score using Poisson Distribution
Let $X_1, X_2, \ldots, X_n \overset{iid}{\sim} \text{Pois}(\lambda)$ with the true vaule of $\lambda$ being $\lambda_{0}$. 
We can thus show that
$$
S(\lambda \mid X_{1} \ , \dots\ , X_{n}) = -n + \frac{\sum_{i} X_{i}}{\lambda}
$$
> Note that the score contains $\sum_{i} X_{i}$ which will change value from one set of sample to the other. It's the sample observation $(X_{1}, X_{2}, \dots, X_{n})$ in the [[Score Function|score function]] which makes it a random variable.

---
## Important Property
Under some assumptions, we can get that
$$
E\left[S(\theta \mid X)\right]\Big|_{\theta=\theta_0} = 0
$$
where the expectation is taken over $X$.

Hence, it follows that
$$
E\left[S(\theta \mid X)\right]\Big|_{\theta=\theta_0} = 0
\implies 
E\left[S(\theta \mid X_{1}, X_{2}, \dots, X_{n})\right]\Big|_{\theta=\theta_0} = 0
$$

---
### Example
Let $X_1, X_2, \ldots, X_n \overset{iid}{\sim} \text{Pois}(\lambda)$ with the true value of $\lambda$ being $\lambda_{0}$.
$$
\begin{aligned}
E\left[S(\lambda \mid X_1, \ldots, X_n)\right] &= E\left[-n + \frac{\sum_i X_i}{\lambda}\right] \\[2ex]
&= -n + \frac{1}{\lambda}\,E\left[\sum_i X_i\right] \\[2ex]
&= -n + \frac{1}{\lambda}\,n\,E[X_i] \\[2ex]
&= -n + \frac{1}{\lambda}\,n\lambda_0
\end{aligned}
$$
Thus, at $\lambda = \lambda_{0}$, we get that $E\left[S(\theta \mid X)\right]\Big|_{\theta=\theta_0} = 0$.

---
## Distribution of Score
Recall that for a single observation $X$, we know that
- [[Score Function|score]] is a random variable
- its expectation is zero
- its variance is $I(\theta_{0})$

> And for multiple observations $X_{1}, \dots,X_{n},$ $S(\theta_{0} \mid X_{1}, \dots, X_{n}) = \sum_{i} S(\theta_{0} \mid X_{i})$ is the sum of $n$ independent random variables with same mean and variance. 

---
### Central Limit Theorem(Different Version)
Suppose $X_{1}, \dots ,X_{n}$ are independently and identically distributed $(iid)$ with $E[X_{i}] = \mu$ and $var[X_{i}] = \sigma^{2}$. Then, 
$$
\sum^{n}_{i=1} X_{i} \overset{D}{\to} N(n\mu, \sigma^{2})
$$
Hence, we get that 
$$
S(\theta_{0} \mid X_{1}, X_{2}, \dots, X_{n}) \overset{D}{\to} N(0, nI(\theta_{0}))
$$

---
