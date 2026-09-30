# Independence between $\bar{X}$ and $S^{2}$
Suppose $X_{1}, X_{2}, \dots, X_{n} \overset{iid}{\sim} \mathbb{N}(\mu, \sigma^{2})$. Then, we have the following theorem:
$$
\boxed{ \ \bar{X} \text{ and } S^{2} \text{ are independent} \ }
$$
where
- $\bar{X} = \frac{1}{n} \sum^{n}_{i=1} X_{i}$ is the [[Estimator|estimator]] of [[Variance|mean]]
- $S^{2} = \frac{1}{n-1} \sum^{n}_{i=1} (X_{i} - \bar{X})^{2}$  is the [[Estimator|unbiased estimator]] of [[Variance|variance]]

---
## Proof
### Preliminary
First, note the following:
$$
\begin{align}
V &= X_{1} - \bar{X} \\[6pt]
&= X_{1} - \frac{1}{n}X_{1} - \frac{1}{n} X_{2} \dots - \frac{1}{n} X_{n}
\\[6pt]
&= \left( 1-\frac{1}{n} \right) X_{1} - \frac{1}{n} X_{2} \dots  
- \frac{1}{n} X_{n}
\end{align}
$$

---
### Independence using Covariance
Using that, we can analyze the [[Covariance|covariance]] between $\bar{X}$ and $X_{1} - \bar{X}$.
$$
\begin{align}
&\text{cov}[\bar{X}, \  X_{1} - \bar{X}] \\[6pt]
= \ &\text{cov}[\bar{X}, \ X_{1}] - \text{cov}[\bar{X}, \bar{X}] \\[6pt]
= \ &\text{cov}\left[ \frac{1}{n} X_{1} + \frac{1}{n}X_{2} + \dots  
+ \frac{1}{n}X_{n}, \ X_{1} \right] - V[\bar{X}] \\[6pt]
= \ &\text{cov}\left[ \frac{1}{n}X_{1}, X_{1} \right]
+ \text{cov}\left[ \frac{1}{n}X_{2}, X_{1} \right] + \ \dots \ - V[\bar{X}]
\\[6pt]
= \ &\frac{1}{n} \text{cov}[X_{1}, X_{1}] + 0 + 0 + \dots - V[\bar{X}]
\\[6pt]
= \ &\frac{1}{n}V[X_{1}] - V[\bar{X}] \\[6pt]
= \ &\frac{1}{n} \sigma^{2} - \frac{\sigma^{2}}{n} = 0
\end{align}
$$

Hence, we get that
$$
\bar{X} \perp\!\!\!\perp ( \ X_{i} - \bar{X} \ )
\quad \text{, for } i=1,2,\dots n
$$

---
### Independence of Function
Functions conserve independence between the random variables.
$$
X \perp\!\!\!\perp Y
\implies f(X) \perp\!\!\!\perp f(Y)
$$
Using that, we can get
$$
\begin{align}
&\bar{X} \perp\!\!\!\perp ( \ X_{i} - \bar{X} \ ) \\[6pt]
&\bar{X} \perp\!\!\!\perp ( \ X_{i} - \bar{X} \ )^{2} \\[6pt]
&\bar{X} \perp\!\!\!\perp \sum( \ X_{i} - \bar{X} \ )^{2} \\[6pt]
\end{align}
$$
since $\bar{X}$ is independent of all the $(X_{i} - \bar{X})^{2}$.

---
### Independence
The [[Sample Variance|sample variance]] $S^{2} = \frac{1}{n-1} \sum^{n}_{i=1}(X_{i} - \bar{X})^{2}$ is a function of all the $(X_{i} - \bar{X})$s.
Hence, $\bar{X}$ is independent of $S^{2}$. $\blacksquare$

---
