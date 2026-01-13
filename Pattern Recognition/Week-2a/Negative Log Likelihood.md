## Negative Log Likelihood
Instead of operating directly on the likelihood function, we will instead optimize the `log likelihood`

$$
\begin{align}
\theta^* 
&= \arg \min_{\theta \in \Theta}
\log \left( \prod^N_{n=1} p(y_{n} \mid \theta) \right) \\[6pt]

&= \arg \max_{\theta \in \Theta}
\sum^N_{n=1} \log(p(y_{n} \mid \theta))
\end{align}
$$

Pragmatically, we often use minimization algorithms, we minimize the `Negative Log Likelihood` as
$$
\theta^* = \arg \min
\sum^N_{n=1} - \log(p(y_{n} \mid \theta))
$$

---
