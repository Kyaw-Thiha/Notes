# Conjugate Priors
A `conjugate prior` is a probability distribution of a specific form such that
> When it is multiplied by the likelihood function, the `posterior` will have the same form as the `prior`

---
`Bernoulli & Beta Distributions`
Let's consider [[Bernoulli Distribution|Bernoulli]] and [[Beta Distribution|Beta]] distributions:
$$
\begin{align}
p(\mu \mid \mathcal{D}, a, b)
&\propto p(\mathcal{D} \mid \mu) 
\ p(\mu \mid a, b) \\[6pt]

&= \left( \prod_{n} \mu^{y_{n}}(1 - \mu)^{1-y_{n}} \right) 
\left( \frac{\Gamma (a+b)}{\Gamma(a) \ \Gamma(b)}
\mu^{a-1} (1-\mu)^{b-1} \right) \\[6pt]

&= \mu^{\#(y_{n}=1)} (1-\mu)^{\#(y_{n}=0)}
\frac{1}{\eta \ \text{Beta}} \mu^{a-1} (1-\mu)^{b-1}
\\[6pt]

&= \frac{1}{\eta \ \text{Beta}}
\mu^{\#(y_{n}=1) + a - 1}
(1-\mu)^{\#(y_{n}=0) + b - 1}
\end{align}
$$

Hence, note that when the prior is `Beta Distribution` for the `Bernoulli Distribution` the updated  distribution remains a `Beta Distribution`.

---
## See Also 
- [[Bernoulli Distribution]]
- [[Beta Distribution]]
