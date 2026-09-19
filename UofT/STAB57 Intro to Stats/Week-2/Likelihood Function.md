# Likelihood Function
Suppose $X_{1}, X_{2}, \dots, X_{n}$ has a joint density 
$$
f(x_{1}, x_{2}, \dots, x_{n} \mid \theta)
$$
Given the following sample of
$$
X_{1} = x_{1} ,  \ X_{2}=x_{2}, \ \dots, 
\ X_{n}=x_{n}
$$
the [[Likelihood Function|likelihood function]] can be defined as
$$
\boxed{ \ L(\theta \mid x_{1}, x_{2}, \dots, x_{n})
= f(x_{1}, x_{2}, \dots, x_{n}\mid \theta) \ }
$$

Note that if $X$ follows a discrete distribution, it gives the probability of observing the sample as a function of the parameter $\theta$.

---
## IID Case
If $X_{1}, \ X_{2}, \ \dots, \ X_{n}$ are $i.i.d$, then their joint density is the product of marginal densities $f_{\theta}(x)$. Hence in $i.i.d$ case, we have
$$
f_{\theta}(x_{1}) + f_{\theta}(x_{2})
+ \dots + f_{\theta}(x_{n})
= \prod^{n}_{i=1} f_{\theta}(x_{i})
$$

---
### Example
Suppose we have a coin of unknown distribution:
$$
\begin{cases}
\ P[H] = \theta & X_{i}=1 \text{ if H} \\[6pt]
\ P[T] = 1-\theta & X_{i}=0 \text{ if T} \\[6pt]
\end{cases}
$$
Suppose we draw the following:
$$
H \quad T \quad H \quad H \quad  T
$$
We could then get
$$
\begin{align}
L(\theta)
&= \theta \times (1 - \theta) \times \theta
\times \theta \times (1 - \theta) \\[6pt]
&= \theta^{3}(1 - \theta)^{2}
\end{align}
$$

**Case-1**: $\theta=\frac{1}{2}$
$$
L\left( \frac{1}{2} \right)
= \left( \frac{1}{2} \right)^{3}
\left( \frac{1}{2} \right)^{2}
= \left( \frac{1}{2} \right)^{5}
$$
$\left( \frac{1}{2} \right)^{5}$ is the probability of observing $\text{HTHHT}$ for a coin of $\theta=\frac{1}{2}$.

**Case-2**: $\theta=0.6$
$$
L(0.6)
= (0.6)^{3}(0.4)^{2}
= 0.034
$$
$0.034$ is the probability of observing $\text{HTHHT}$ for a coin of $\theta=0.6$.

**Bernoulli Distribution**: This coin has [[Bernoulli Distribution|Bernoulli Distribution]]. 
In general for $n$ samples, we have
$$
L(\theta \mid x_{1}, x_{2}, \dots, x_{n})
= \theta^{\left( \sum^{n}_{i=1} x_{i} \right)}
(1-\theta)^{\left(n - \sum^{n}_{i=1} x_{i} \right)}
$$

---
## Notes
- $L(\theta)$ is NOT a pdf or pmf of $\theta$.
- For $\theta_{1}, \theta_{2} \in \Omega$, we believe in $\theta_{1}$ as true value of $\theta$ whenever 

$$
L(\theta_{1}) > L(\theta_{2})
$$

- The value $L(\theta)$ is very small for every value of $\theta$. So, we are often interested in the likelihood ratio:

$$
\frac{L(\theta_{1})}{L(\theta_{2})}
$$

---