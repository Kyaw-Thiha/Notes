### Maximum Likelihood Estimation (MLE)
#ml/statistical-learning/mle
The `maximum likelihood estimator` optimizes the likelihood function in order to find the parameters:
$$
\theta^* = \arg \max_{\theta \in \Theta} \ 
p(\mathcal{D} \mid \theta)
$$

This assumes that distribution is jointly over all values of $y_{n} \in \mathcal{D}$.

We can simplify this by assuming $i.i.d$:

$$
\theta^* = \arg \max_{\theta \in \Theta} \
\prod^N_{n=1} p(y_{n} \mid \theta)
$$
But note that $\prod^N_{n=1} \ p(y_{n} \mid \theta)$ is the product of many numbers less than $1$.
Hence, the final value will be very small.

---