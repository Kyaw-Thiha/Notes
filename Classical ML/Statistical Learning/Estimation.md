# Estimation

We can use `Estimation` methods to find the best model parameters (point estimate)

### `MAP (Maximum A Posteriori)`
$$
\begin{align}
\hat{\delta}_{MAP}  
&= argmax_{\theta} ( P(\theta|D) ) \\[6pt]
&= argmax_{\theta}(P(D|\theta).P(\theta))
\end{align}
$$

### `MLE (Maximum Likelihood)`
$$
\hat{\theta}_{ML} = argmax_{\theta} (P(D|\theta) )
$$


If the `prior` is very uninformative, we can consider it as uniformly distributed.
Hence, $P(\theta)$ is constant.
Thus, $P(\theta|D) \propto P(D|\theta).P(\theta)$ which means `MLE` can be considered as special case of `MAP`.

Also, note that `MLE` tends to overfit more.

To calculate these, we use the [[Log Likelihood]].

## See Also
- [[MAP for Binomial Distribution]]
- [[MLE for Gaussian Distribution]]
- [[MAP for Non-Linear Regression]]