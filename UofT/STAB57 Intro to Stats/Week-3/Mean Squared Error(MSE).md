# Mean Squared Error
[[Mean Squared Error(MSE)|Mean-squared error]] measures the average of the squares of error.
$$
\text{MSE}_{\theta}(T) = \mathbb{E}_{\theta}[(T - \theta)^{2}]
$$
where
- $\theta$ is the unknown parameter
- $T$ is an [[Estimator|estimator]] of $\theta$

The smaller the value of $\text{MSE}_{\theta}(T)$ is, the more concentrated the sampling distribution of $T$ is about the value of $\theta$.

---
## Bias-Variance Decomposition

![image|300](https://notes-media.kthiha.com/Mean-Squared-Error(MSE)/30acccf839f953f06084a52ad176abae.png)
$$
\boxed{\text{MSE}(T) = \text{var[T]} + (\mathbb{E}[T] - \theta)^{2}}
$$


### Proof
First, recall that
$$
V[T- \theta] = \mathbb{E}[(T - \theta)^{2}] - (\mathbb{E}[(T - \theta)])^{2}
$$
Using that, we get
$$
\begin{align}
V[T] &= \text{MSE}[T] - (E[T] - E[\theta])^{2} \\[6pt]
&= \text{MSE}[T] - (E[T] - \theta)^{2} \\[6pt]
&= \text{MSE}[T] - (\text{Bias}[T])^{2} \\[6pt]
\end{align}
$$
This implies that
$$
\text{MSE}[T] = V[T] + (\text{Bias}[T])^{2}
$$

Read more about [[Bias-Variance Decomposition]].

---
## Bias
The bias of an [[Estimator|estimator]] $T$ of $\theta$ is the difference between $E[T]$ and $\theta$:
$$
\text{Bias}[T] = E[T] - \theta
$$

---
## Unbiased Estimator
When the [[Bias-Variance Decomposition|bias]] of an [[Estimator|estimator]] is zero, it's called unbiased. 
- So $T$ is unbiased estimator of $\theta$ when 
	- $E[T]=\theta$
	- $\text{MSE}[T] = \text{var}[T]$
- In other words, $T$ is unbiased if $\theta$ is the mean of the sampling distribution of $T$.

---
