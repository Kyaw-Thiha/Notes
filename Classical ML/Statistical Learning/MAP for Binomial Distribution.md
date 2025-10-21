# MAP for Binomial Distribution
We flip a coin 𝑁 times, and we got 𝐾 heads. 
Assume coin flips are independent and the probability of the coin to land on a head is 𝜃 and we believe all values of 𝜃 are equally likely a priori, what would be $\hat{\theta}_{MAP}$?

- A coin flip outcome: $c_i \in \{H, T\}$  
- A sequence of all the $N$ coin flips: $c_{1:N} = c_1, c_2, \ldots, c_N$  
- Probability of landing on a head: $p(c = H) = \theta$  
- Probability of landing on a tail: $p(c = T) = 1 - \theta$  
- Prior of $\theta$: $\theta \sim \mathcal{U}[0, 1]$, uniformly distributed between $[0,1]$

The `likelihood` function is the probability of observing $K$ heads out of $N$ flips, given $\theta$:
$$
P(D | \theta) = P(c_{1:N} | \theta) = \theta^K(1 - \theta)^{N-K}
$$

As given, the `prior` is uniform
$$
P(\theta) = 1, \text{ for } \theta \in [0, 1]
$$

Hence,
$$
\text{Likelihood} \times \text{Prior} = P(D | \theta) \times P(\theta) = \theta^K(1 - \theta)^{N-K}
$$

Getting the `MAP`,
$$
\begin{align}
\hat{\theta}_{MAP}
&= argmin_{\theta} [-\ln(P(D|\theta).P(\theta))] \\[6pt]
&= argmin_{\theta} [-\ln(\theta^K(1 - \theta)^{N-K})] \\[6pt]
&= argmin_{\theta} [-K.\ln(\theta) - (N-K).\ln(1 - \theta)]
\end{align}
$$

Let $L(\theta) = -K.\log(\theta) - (N-K).\log(1 - \theta)$
Then,
$$
\begin{align}
\frac{dL(\theta)}{d\theta} &= 0 \\[6pt]
-\frac{K}{\theta} + (N-K). \frac{1}{1 - \theta} &= 0 \\[6pt]
\end{align}
$$

This implies that
$$
\hat{\theta}_{MAP} = \frac{K}{N}
$$
