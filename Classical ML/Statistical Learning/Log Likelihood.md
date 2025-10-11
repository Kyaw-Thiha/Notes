# Log Likelihood

When computing [[Estimation]], we work with `log posterior` and `log likelihood` functions due to numerical reasons, specifically:

- Since probability is $[0, 1]$, $\log$ help prevent computing with very small data.
- $\log$ has good derivative
- $\log$ keeps the big values smaller

Continuing from [[Estimation]], we get
$$
\begin{align}
\hat{\theta}_{MAP}   
&= argmax_{\theta}.P(\theta | D) \\[6pt]
&= argmax_{\theta} P(D | \theta).P(\theta) \\[6pt]
&= argmax_{\theta} \log(P(D|\theta).P(\theta)) \\[6pt]
&= argmax_{\theta} [\log(P(D|\theta)) + \log(P(\theta)) ] \\[6pt]
\end{align}
$$

To minimize, we can convert it into negative like
$$ 
\hat{\theta}_{MAP} = argmax_{\theta} [- \log(P(D|\theta)) - \log(P(\theta)) ] \\[6pt]
$$
