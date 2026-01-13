### Rules of Probability Review
- `Sum Rule`: $p(X) = \sum_{Y} p(X,Y)$
- `Product Rule`: $p(X,Y) = p(Y|X)p(X) = P(X|Y)p(Y)$
- `Independence`: $p(X,Y) = p(Y)p(X) = p(X)p(Y)$

[[Bayes Rule]] follows from `sum rule` and `product rule.`
$$
\begin{align}
&p(y|x) \ p(x) = p(x|y)\ p(y) \\[6pt]
\implies &p(y|x) = \frac{p(x|y) p(y)}{p(x)}
\end{align}
$$
---
### Bayesian Probability
Let dataset $\mathcal{D} = ((x_{1}, y_{1}), \ \dots, \ (x_{N}, y_{N}))$.
Suppose we want to find $\hat{y} = h(x; \theta)$.
What more do we need than prediction error?

---
`Frequentist Probability`
It allows us to capture the uncertainty in our predictions.
$$
p(\mathcal{D} \ | \ \theta) \ \quad 
\text{the likelihood function}
$$
The shape of distribution $p(\mathcal{D} \mid \theta)$ lets us capture how much we care about deviations between $f(x)$ and $h(x; \ \theta)$.

---
`Bayesian Probability`
$$
p(\theta \mid \mathcal{D}) 
= \frac{p(\mathcal{D} \mid \theta) \ p(\theta)}{p(\mathcal{D})}
\ \quad \text{the posterior distribution}
$$
Where
- $p(\theta)$: prior belief over the values $\theta$ can take on
- $p(\theta \mid \mathcal{D})$: posterior belief over values of $\theta$, given the data $\mathcal{D}$

---
### Negative Log Likelihood
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
## MLE & MAP
To optimize the likelihood function, we can use [[Maximum Likelihood Estimation (MLE)|MLE]]
$$
\theta^* = \arg \max_{\theta \in \Theta} \ 
p(\mathcal{D} \mid \theta)
$$

If we want to model a prior, we can use [[Maximum A Posteriori (MAP)|MAP]]
$$
\begin{align}
\theta^*
&= \arg \max_{\theta \in \Theta}
\ p(\theta \mid \mathcal{D}) \\[6pt]

&= \arg \max_{\theta \in \Theta}
\frac{p(\mathcal{D} \mid \theta) \ p(\theta)} 
{\mathcal{D}}
\end{align}
$$
Optimizing the [[Negative Log Likelihood]], we can get
$$
\theta^* 
= \arg \min_{\theta \in \Theta}
- \sum^N_{n=1} \log(p(y_{n} \mid \theta))
- \log(p(\theta)) + \log(p(\mathcal{D}))
$$

See example with [[Bernoulli Distribution]].

---
### Conjugate Priors
A `conjugate prior` is a probability distribution of a specific form such that
> When it is multiplied by the likelihood function, the `posterior` will have the same form as the `prior`

[[Conjugate Priors|Read More]]

---
