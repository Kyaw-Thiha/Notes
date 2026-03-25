# Sample Average Estimator
#rl/online-learning
The `sample average estimator` is defined as
$$
\boxed{ \ m_{t} \triangleq \frac{1}{t} 
\sum ^{t}_{i=1} Z_{i} \ }
$$
---
## Motivation
> Consider a simple problem of estimating the mean of a random variable, given samples from it.

Concretely, assume we are given $n$ real-valued $r.v$ $Z_{1}, \ \dots, \ Z_{n}$ all drawn $i.i.d$ from a distribution $v$.
How can we estimate expectation $m = \mathbb{E}\ [\mathbb{Z}]$ with $\mathbb{Z} \sim v$?

We could use the `sample average`
$$
m_{t} \triangleq \frac{1}{t} \sum^t_{i=1} Z_{t}
$$
Under mild conditions by the `Law of Large Numbers`, $m_{t} \to m$.


---
## Analysis on Mean
> The variable $m_{t}$ is a random variable and concentrates around its expectation.
> Hence, it is an `unbiased estimator`.

Let the expectation of $Z_{i} \sim v$ be $\mu$.
Let the variance be $\sigma^{2}$.
By the linearity of expectation, we get
$$
\mathbb{E}[\ m_{t} \ ]
= \mathbb{E}\left[ \frac{1}{t} \sum ^{t}_{i=1} Z_{i} \right]
= \frac{1}{t} \sum^{t}_{i=1} \mathbb{E}[Z_{i}]
= \frac{1}{t} \ \mathbb{E}[\mu]
= \mu
$$

---
## Analysis on Variance
> Variance is dispersion of a random variable around its mean.
> Hence, a decreasing variance show that $m_{t}$ is increasingly more concentrated around $\mu$.

Using the independence of $Z_{i}$ and $Z_{j}$, we get
$$
\begin{align}
Var[\ m_{t} \ ]
&= \mathbb{E}[(m_{t} - \mathbb{E}[m_{t}])^{2}]  
\\[6pt]

&= \mathbb{E}\left[ \left( \frac{1}{t}  
\sum^{t}_{i=1} (Z_{i} - \mu) \right)^{2} \right]
\\[6pt]

&= \frac{1}{t^{2}} \ \mathbb{E} \left[  
\sum^{t}_{i,j=1} (Z_{i} - \mu)(Z_{j} - \mu) \right]
\\[6pt]

&= \frac{1}{t^{2}} \ \mathbb{E} \left[  
\sum^{t}_{i=1} (Z_{i} - \mu)(Z_{i} - \mu)  
+ \sum^{t}_{i,j=1, \ i\neq j} (Z_{i} - \mu) 
(Z_{j} - \mu) \right]
\\[6pt]

&= \frac{1}{t^{2}} \ \mathbb{E} \left[  
\sum^{t}_{i=1} \sigma^{2}  
+ \sum^{t}_{i,j=1, \ i\neq j} (Z_{i} - \mu) 
(Z_{j} - \mu) \right]
\\[6pt]

&= \frac{\sigma^{2}}{t} + \frac{1}{t^{2}}
\sum^{t}_{i,j=1, \ i\neq j} \mathbb{E}[(Z_{i} - \mu)]
\ \mathbb{E}[(Z_{j} - \mu)] \\[6pt]

&= \frac{\sigma^{2}}{t}
\end{align}
$$

This shows that as $t$ increases, the variance of $m_{t}$ decreases with a rate of $\frac{1}{t}$.


---
## Weak Law of Large Number (LLN)
For any $\epsilon > 0$ as $t\to \infty$,
$$
\boxed{ \ \lim_{ t \to \infty } \mathbb{P} 
\{ |m_{t} - \mu| > \epsilon \} \to 0 \ }
$$
This is the convergence in probability of $m_{t}$ to $\mu$.
This result is also known as [[Weak Law of Large Numbers (LLN)|weak law of large numbers]].

---
### Proof
Recall that 
- $\mathbb{E}[m_{t}] = \mu$
- $\text{Var}[m_{t}] = \frac{\sigma^{2}}{t}$

Applying [[Markov's Inequality]] to the non-negative $r.v.$ $Z=|m_{t} - \mu|^{2}$,
$$
\begin{align}
\mathbb{P}\{ |m_{t} - \mu| > \epsilon \} 
&= \mathbb{P} \{ |m_{t} - \mu|^{2} > \epsilon^{2} \}
\\[6pt]
&\leq \frac{\mathbb{E}[ \ |m_{t} - \mu|^{2} \ ]}{\epsilon^{2}} \\[6pt]
&= \frac{\text{Var}[ \ m_{t} \ ]}{\epsilon^{2}}  
\\[6pt]
&= \frac{\sigma^{2}}{t \ \epsilon^{2}}
\end{align}
$$
This shows that for any $\epsilon>0$ as $t\to \infty$,
$$
\lim_{ t \to \infty } \mathbb{P} 
\{ |m_{t} - \mu| > \epsilon \} \to 0 
$$
> Hence, the probability that $m_{t}$ is more than $\epsilon$ different from $\mu$ is `asymptotically zero` no matter how small $\epsilon$ is.

---
## Strong Law of Large Number
We also have `strong LLN` which states that
$$
m_{t} \to \mu
\quad \text{almost surely}
$$
under mild assumptions such as $\mathbb{E}[\ |Z_{i}|\ ] < \infty$ for all $i$.

---
## See Also 
- [[Markov's Inequality]]
- [[Online Estimator of Mean of Random Variable]]
- [[Weak Law of Large Numbers (LLN)]]