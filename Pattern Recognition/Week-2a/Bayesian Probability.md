## Bayesian Probability
#ml/statistical-learning/bayesian-probability
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
