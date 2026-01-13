## Bernoulli Distribution
The `Bernoulli distribution` models as
$$
\text{Bern}(y \mid \mu) 
= \mu^y \ (1 - \mu)^{1-y}
$$
It has 
- Mean $\mathbb{E}[t] = \mu$
- $var(t) = \mu(1 - \mu)$

With many observation, we get
$$
\begin{align}
p(\mathcal{D} \mid \mu)
&= \prod^N_{i=1} p(y_{n} \mid \mu) \\[6pt]
&= \prod^N_{i=1} \mu^{y_{n}} \ (1 - \mu)^{1 - y_{n}}
\end{align}
$$
And assuming $i.i.d$, we get
$$
\log(p(\mathcal{D} \mid \mu))
= \sum^N_{n=1} y_{n} \log \mu
+ (1 - y_{n}) \ \log(1 - \mu)
$$

---
`Finding MLE of Bernoulli Distribution`
Carrying $\frac{d}{dx} [\log(x)]$ through,
$$
\begin{align}
\frac{d}{d\mu}
\ \log p(\mathcal{D} \mid \mu)
&= \sum^N_{n=1} y_{n} \frac{1}{\mu}  
- (1- y_{n}) \frac{1}{ 1 - \mu}
\end{align}
$$

Optimizing it, we get
$$
\begin{align}
0 &= - \sum^N_{n=1} \frac{1}{1-\mu}
+ \sum^N_{n=1} y_{n} \left( \frac{1}{\mu} + \frac{1}{1 - \mu} \right) \\[6pt]

\frac{N}{1-\mu} &= \frac{1 - \mu + \mu}{\mu(1-\mu)}
\ \sum^N_{n=1} y_{n}   
\ \quad \text{where } \sum^N_{i=1} i = N, \text{ rearrange} \\[6pt]

\mu &= \frac{1}{N} \sum^N_{n=1} y_{n}
\end{align}
$$

---
`Finding mAP of a Bernoulli Distribution`
Let $\text{prior } \mu$ be $p(\mu) \sim \mathcal{N}(\gamma, \sigma^2)$.
Using [[Maximum A Posteriori (MAP)|MAP]], we get
$$
\arg \min_{\theta, \sigma^2 \in \Theta}
- \sum^N_{n=1} \log(p(y_{n} \mid \mu))
- \log\left( \exp\left( -\frac{1}{2} \frac{(\mu - \theta)^2}{\sigma^2} \right) \right)
$$

By carefully picking the prior distribution, we can it easier.
This is considered choosing from a class of [[Conjugate Priors]].

---
