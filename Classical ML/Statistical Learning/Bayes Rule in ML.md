# Bayes Rule in ML
#ml/bayes-rule

[[Bayes Rule]] can be applied to machine learning as statistically learning to better under models.

$$
p(M | D) = \frac{p(D|M).p(M) }{p(D)}
$$

`Posterior Distribution` $P(M | D)$: Distribution over model parameters after seeing the data.

`Prior Distribution` $P(M)$: Distribution over model parameters that we assume before seeing the data.

`Likelihood` $P(D | M)$: Consistency of data with the model. This is function of model; not data.

`Evidence` $P(D)$: Marginal likelihood of data over all possible models.

### Obtaining Posterior

By `Bayes Rule`, we can deduce that $Posterior \propto \text{Likelihood} \times \text{Prior}$

$$
P(M|D) \propto P(D|M) \times P(M)
$$

This mean that to obtain `Posterior`, we need
- Data to form `likelihood` $P(D|M)$
- A `prior` probability distribution $P(M)$ to represent our prior believe.
