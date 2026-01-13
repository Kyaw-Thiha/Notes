# Beta Distribution
The `beta distribution` is defined over quantities $\mu \in [0,1]$.
It depends on 2 parameters, $a$ and $b$ and is defined as:
$$
\text{Beta}(\mu \mid a,b)
= \frac{\Gamma(a + b)}{\Gamma(a) \ \Gamma(b)}
\mu^{a-1} \ (1 - \mu)^{b-1}
$$

Note that $\frac{\Gamma(a+b)}{\Gamma(a) \ \Gamma(b)}$ is a `normalizing constant`.

> Being defined over $\mu \in [0,1]$ makes it a good choice as a prior for [[Bernoulli Distribution]].

---
`Mean & Variance`

Beta Distribution has
- $\mathbb{E}[\mu] = \frac{a}{a+b}$ and,
- $\text{Var}[\mu] = \frac{ab}{(a+b)^2 \ (a+b+1)}$

---
## See Also
- [[Bernoulli Distribution]]