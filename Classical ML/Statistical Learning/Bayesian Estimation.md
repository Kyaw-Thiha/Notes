# Bayesian Estimation

`MAP Estimation` finds the mode of the posterior distribution.
$$
\hat{\theta}_{MAP} = argmax_{\theta} \ p(\theta | D)
$$

`Bayesian Estimation` finds the mean of the posterior distribution.

$$
\hat{\theta}_{Bayes} = E[\theta | D] = \int \theta \  p(\theta|D)  \ d\theta
$$

## Bernoulli Example
For example, for `Bernoulli Distribution`, 
`MAP Estimation`: $\frac{K}{N}$
`Bayesian Estimation`: $\frac{K+1}{N+1}$

This means that `Bayesian Estimate` is biased towards $\frac{1}{2}$ compared to `MAP`

## Gaussian Example
For example for `Gaussian Distribution`,
`MAP` and `Bayesian Estimation` are the same
