# Consistent Estimator
> Let $T_{n}$ be an [[Estimator|estimator]] of parameter $\theta$. $T_{n}$ is said to be [[Consistent Estimator|consistent(in probability)]] if $T_n \xrightarrow{P} \theta$. 
> In other words, $T_{n}$ converges to $\theta$ in probability.

![image|300](https://notes-media.kthiha.com/Consistent-Estimator/de4a134244d988e2ec7144c172363586.png)

---
## Proving Consistency using LLN
[[Law of Large Numbers(LLN)|LLN]] tells us that $\bar{X} = \frac{1}{n}\sum X_i \xrightarrow{P} E[X_i]$ for any distribution.
Hence, we can deduce that
- If $X_i \overset{iid}{\sim} N(\mu, \sigma^2)$ then $\bar{X}$ is a consistent estimator of $\mu$
- If $X_i \overset{iid}{\sim} \text{Poisson}(\lambda)$ then $\bar{X}$ is a consistent estimator of $\lambda$
- And we can say this for a few other known distributions

To prove [[Consistent Estimator|consistency]] when the [[Estimator|estimator]] is not simpy $\bar{X}$, we can 
- still use LLN, but with the help of
- [[#Slutsky's Lemma|Slutsky's Lemma]]
- [[#Continuous Mapping Theorem|Continuous Mapping Theorem]]

---
### Slutsky's Lemma
We have two different sequences $X_{n}$ and $Y_{n}$.
- $X_n \xrightarrow{P} X \;\text{ and }\; Y_n \xrightarrow{P} Y \;\Longrightarrow\; X_n + Y_n \xrightarrow{P} X + Y$
- $X_n \xrightarrow{P} X \;\text{ and }\; Y_n \xrightarrow{P} Y \;\Longrightarrow\; X_n Y_n \xrightarrow{P} XY$

---
### Continuous Mapping Theorem
> Let $X_{n} \xrightarrow{P} X$ and $g(\cdot)$ be a continuous function.
> Then, $g(X_{n}) \xrightarrow{P} g(X)$.

---
### Proving $S^{2}$ is a Consistent Estimator of $\sigma^{2}$
Using [[#Continuous Mapping Theorem|continuous mapping theorem]], we can say that $S$ is a [[Consistent Estimator|consistent estimator]] of $\sigma$.
$$
\begin{aligned}
S^2 &= \frac{1}{n-1}\sum \left(x_i - \bar{x}\right)^2 \\[2ex]
&= \frac{n}{n-1} \cdot \frac{1}{n}\left(\sum x_i^2 - n\bar{x}^2\right) \\[2ex]
&= \frac{n}{n-1} \cdot \left\{\underbrace{\frac{1}{n}\sum x_i^2}_{\xrightarrow{P}\, E[X_i^2]} - \underbrace{\bar{x}^2}_{\xrightarrow{P}\, (E[X_i])^2}\right\} \\[2ex]
&= \underbrace{\frac{n}{n-1}}_{\to\, 1} \cdot \underbrace{\left\{E[X_i^2] - \left(E[X_i]\right)^2\right\}}_{=\,\sigma^2} \\[2ex]
&= 1 \cdot \sigma^2
\end{aligned}
$$

---
## MLE Consistent
An [[Estimator|estimator]] $T_{n}$ is [[#MLE Consistent|MLE consistent]] if
$$
\boxed{MSE(T_n) \to 0 \text{ as } n \to \infty}
$$
In other words, after [[Maximum Likelihood Estimation(MLE)|MLE]] of an estimator is calculated, check if it goes to zero for large $n$.

Note that [[#MLE Consistent|MLE Consistent]] implies [[Consistent Estimator|consistency in probability]], and is often used in practice.

---
## Proof of MLE Consistent
### Proof Setup
Suppose the following
- $\theta_0$ : The TRUE value of the parameter which produced the data. (which is an unknown constant)
- Suppose $(X_1, X_2, \ldots, X_n) \overset{iid}{\sim} f(x \mid \theta_0)$
- $\hat{\theta}$ is the MLE

> **Claim**
> $\hat{\theta}$ converges to $\theta_0$ in probability $\left(\hat{\theta} \xrightarrow{P} \theta_0\right)$.

---
### Proof (using Illustration)
Let us consider the situation where we haven’t observed the samples yet. We have the [[Log Likelihood|log likelihood]] of 
$$
\ell(\theta) = \sum_{i=1}^{n} \log f(X_i \mid \theta)
$$
In other words, the [[Log Likelihood|log likelihood function]] varies from one set of sample to another. Dividing both sides by sample size $n$, we get
$$
\frac{1}{n}\ell(\theta) = \frac{1}{n}\sum_{i=1}^{n} \log f(X_i \mid \theta)
$$
Note that the right hand side of the above equation is a sample mean. 
Applying [[Law of Large Numbers(LLN)|LLN]], we get
$$
\frac{1}{n}\sum_{i=1}^{n} \log f(X_i \mid \theta) \xrightarrow{P} E\left[\log f(X_i \mid \theta)\right]
$$
Since $\frac{1}{n}\ell(\theta)$ gets closer to $E\left[\log f(X_i \mid \theta)\right]$, the $\theta$ that maximizes $\frac{1}{n}\ell(\theta)$ should be close to the $\theta$ that maximizes $E\left[\log f(X_i \mid \theta)\right]$

![image|300](https://notes-media.kthiha.com/Consistent-Estimator/6246d7620b29b759f9a2003a8d6bcccf.png)

Hence, we have that
$$
E\left[\log f(X_i \mid \theta)\right] = \int_x \log f(x \mid \theta)\, f(x \mid \theta_0)\, dx
$$
Now we need to show that $E\left[\log f(X_i \mid \theta)\right] \text{ is maximized at } \theta_0$.

---
### Maximization at $\theta_{0}$
>We want to show $\theta_0$ maximizes $E\left[\log f(x_i \mid \theta)\right]$.
> We want to show $\theta_0$ maximizes $\displaystyle\int_x \log f(x_i \mid \theta)\, f(x_i \mid \theta_0)\, dx$.

First, we have that
$$
\begin{aligned}
&\frac{d}{d\theta}\int_x \log f(x_i \mid \theta) \cdot f(x_i \mid \theta_0)\, dx \\[2ex]
\Rightarrow\quad &\int_x \bbox[#CFE2F3,5px]{\color{black}\frac{d}{d\theta}\log f(x_i \mid \theta)} \cdot f(x_i \mid \theta_0)\, dx \\[2ex]
\Rightarrow\quad &\int_x \bbox[#CFE2F3,5px]{\color{black}\frac{1}{f(x_i \mid \theta)} \cdot \frac{d}{d\theta} f(x_i \mid \theta)} \cdot f(x_i \mid \theta_0)\, dx
\end{aligned}
$$
At $\theta = \theta_{0},$ we have
$$
\begin{aligned} 
&\int_x \cancel{\frac{1}{f(x_i \mid \theta_0)}} \cdot \frac{d}{d\theta} f(x_i \mid \theta_0) \cdot \cancel{f(x_i \mid \theta_0)}\, dx \\[2ex]
\implies\quad & \int_x \frac{d}{d\theta} f(x_i \mid \theta_0)\, dx \\[2ex] 
\implies \quad & \frac{d}{d\theta}\int_x f(x_i \mid \theta_0)\, dx \qquad \text{(swap derivative and integral)} \\[2ex] 
\implies\quad & \frac{d}{d\theta}\, 1 \\[2ex] 
\implies\quad & 0 \end{aligned}
$$

---
