# Stochastic Approximation Analysis
#rl/stochastic-approximation
An example of [[Stochastic Approximation (SA) Analysis|stochastic approximation methods]] would be [[Online Estimator of Mean of Random Variable]] denoted as
$$
\theta_{t+1} = (1 - \alpha_{t}) \theta_{t}
+ \alpha_{t} Z_{t}
$$
where $\theta_{t}$ is a random variable.

Hence, we have a various choices of $\alpha_{t}$
- if $\alpha_{t} = \frac{1}{t+1}$, we get the [[Sample Average Estimator|sample average estimator]]
- fixed $\alpha_{t} = a$
- $a_{t} = \frac{c}{t^{p} + 1}$

---
## Studying fixed $\alpha_{t}$

Let $\alpha_{t}$ be a fixed $\alpha$. Then,
$$
\theta_{t+1} = (1 - \alpha) \theta_{t}
+ \alpha Z_{t}
$$
We shall be studying its expectation and variance.

---
### Expectation
Taking expectation over both sides, we get
$$
\begin{align}
\mathbb{E}[\theta_{t+1}]
&= \mathbb{E}[(1 - \alpha) \theta_{t} + \alpha Z_{t}]
\\[6pt]
&= (1 - \alpha) \ \mathbb{E}[\theta_{t}]
+ \alpha \ \mathbb{E}[Z_{t}] \\[6pt]
&= (1 - \alpha) \ \mathbb{E}[\theta_{t}] 
+ \alpha m
\end{align}
$$

Let's denote $\mathbb{E}[\theta_{t}]$ by $\bar{\theta}_{t}$ (which is not a $r.v.$ anymore).
Then, we can rewrite the equation as
$$
\bar{\theta}_{t+1}
= (1 - \alpha) \ \bar{\theta}_{t} + \alpha \ m
$$

Now let's observe $\bar{\theta}_{t}$ as $t$ increases.
Assuming that $\theta_{0} = 0$ $( \implies \bar{\theta}_{0} = 0)$  and $0<\alpha<1$, we get
$$
\begin{align}
&\bar{\theta}_{1} = \alpha m \\[6pt]
&\bar{\theta}_{2} = (1 - \alpha) \ \alpha m  
+ \alpha m  \\[6pt]
&\bar{\theta}_{3} = (1 - \alpha)^{2} \ \alpha m  
+ (1 - \alpha) \ \alpha m + \alpha m  \\[6pt]
& \ \vdots \\[6pt]
&\bar{\theta}_{t} = \alpha \sum^{t-1}_{i=0}
(1 - \alpha)^{i} \ m
= \frac{\alpha m (1 - (1 - \alpha)^{t})} 
{1 - (1 - \alpha)} 
= m [ 1 - (1- \alpha)^{t} ]
\end{align}
$$

Hence, we get that
$$
\bar{\theta}_{t}
= m [ 1 - (1- \alpha)^{t} ]
\implies \lim_{ t \to \infty } \bar{\theta}_{t} = m
$$
Therefore, we can conclude that
- $\theta_{t}$ converges to $m$ in expectation.
- But this is not enough.
  There could be a large deviation around mean at convergence.

---
### Variance
Using independence of $Z_{t}$:
$$
\begin{align}
\text{Var}[\theta_{t+1}]
&= \text{Var}[ \ (1 - \alpha) \ \theta_{t}  
+ \alpha Z_{t} \ ] \\[6pt]
&= (1 - \alpha)^{2} \ \text{Var}[\theta_{t}]
+ \alpha^{2} \ \text{Var}[Z_{t}]
\end{align}
$$
Hence, we have that
$$
\text{Var}[\theta_{t+1}]
\geq \alpha^{2} \text{Var}[Z_{t}]
= \alpha^{2} \sigma^{2}
$$
Therefore for a constant $\alpha$, the variance of $\theta_{t}$ is not going to converge to zero. Instead, $\theta_{t}$ fluctuates around its mean.

For an exact analysis, let $\beta=(1-\alpha)^{2}$ and $U_{t} = \text{Var}[\theta_{t}]$.
Hence, we have that
$$
\begin{align}
&U_{0} = 0 \\[6pt]
&U_{1} = \alpha^{2} \sigma^{2} \\[6pt]
&U_{2} = \alpha^{2} \sigma^{2} \ (1 - \beta) \\[6pt]
& \ \vdots \\[6pt]
&U_{t} = \alpha^{2}\sigma^{2} \sum^{t-1}_{i=0}
\beta^{i}
= \frac{\alpha^{2}\sigma^{2} (1 - \beta^{t})} {1 - \beta} \\
&= \frac{\alpha\sigma^{2} \ (1 - \beta^{t})} 
{1 - \beta}
= \frac{\alpha \sigma^{2}[1 - (1 - \alpha)^{2t}]} 
{2 - \alpha}
\end{align}
$$
So, we get that
$$
\lim_{ t \to \infty } \text{Var}[\theta_{t}] 
= \frac{\alpha \sigma^{2}}{2 - \alpha}
$$

Hence, we can conclude the following
- for a constant $\alpha$, the variance of $\theta_{t}$ is not going to converge to zero
- $\theta_{t}$ fluctuates around its mean
- In order to make $\theta_{t}$ converge in a stronger sense than expectation, we need $\alpha_{t} \to 0$ with some schedule
- $\alpha_{t} = \frac{1}{t+1}$ works, but it is not the only acceptable one
- But not every sequence where $\alpha_{t}$ goes to zero work.
  If it converges too fast, it would not allow enough adaptation.

---
## Stochastic Approximation Conditions
Based on the above analysis, we get the following conditions for [[Stochastic Approximation(SA)|stochastic approximation]]:
- $\sum ^{\infty}_{t=0} \alpha_{t} = \infty$
- $\sum ^{\infty}_{t=0} \alpha_{t}^{2} < \infty$

---
## See Also
- [[Sample Average Estimator]]
- [[Stochastic Approximation(SA)]]