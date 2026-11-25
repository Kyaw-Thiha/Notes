# Cramer-Rao Inequality
> Let $X_{1}, X_{2}, \dots, X_{n}$ be $iid$ with density of $f_{\theta_{0}}(x)$. 
> Then, we have that $T=t(X_{1}, X_{2}, \dots, X_{n})$ be an [[Estimator|unbiased estimator]] of $\theta_{0}$. 

Then, under some assumptions of $f_{\theta_{0}}(x)$, we have that
$$
var[T] \geq \frac{1}{nI(\theta_{0})}
$$
where $\frac{1}{nI(\theta_{0})}$ is known as the [[Cramer-Rao Inequality|Cramer-Rao lower bound(CRLB)]].

---
## Proof of Cramér-Rao inequality

Let $T=t(X_1,\ldots,X_n)$ be an unbiased estimator of $\theta$, so $E[T]=\theta$.

Let $Z$ be the score evaluated at $\theta=\theta_0$

$$
Z = S(\theta\mid X_1,X_2,\ldots,X_n)\Big|_{\theta=\theta_0} = \sum_{i=1}^n \frac{\partial}{\partial\theta}\log f(X_i\mid\theta)\Big|_{\theta=\theta_0}
$$

Immediately we can write, $E[Z]=0$ and $var[Z]=nI(\theta_0)$.

---
### Correlation bound

Correlation coefficient between two variables $T$ and $Z$ is defined as

$$
\rho[T,Z] = \frac{cov[T,Z]}{\sqrt{var[T]\,var[Z]}}
$$

which is bounded between $-1$ and $1$. Then,

$$
\begin{aligned}
-1 \le \rho[T,Z] &\le 1 \\[6pt]
\implies \rho^2[T,Z] &\le 1 \\[6pt]
\implies \left\{\frac{cov[T,Z]}{\sqrt{var[T]\,var[Z]}}\right\}^2 &\le 1 \\[6pt]
\implies \frac{\{cov[T,Z]\}^2}{var[T]\,var[Z]} &\le 1 \\[6pt]
\implies var[T] &\ge \frac{\{cov[T,Z]\}^2}{var[Z]}
\end{aligned}
$$

Continuing from the last step, using $var[Z]=nI(\theta_0)$,

$$
\implies var[T] \ge \frac{\{cov[T,Z]\}^2}{nI(\theta_0)}
$$

We just need to show that $cov[T,Z]=1$.

---
### Showing $cov[T,Z]=1$

Since $E[Z]=0$,

$$
cov[T,Z] = E[TZ] - E[T]\underbrace{E[Z]}_{=0} = E[TZ]
$$

Writing the expectation as an integral over the joint pdf, with $\prod f(x_i\mid\theta)$ as the pdf,

$$
E[TZ] = \int_{x_1}\int_{x_2}\cdots\int_{x_n} \underbrace{t(x_1,\ldots,x_n)}_{T}\;\underbrace{\sum\frac{\partial}{\partial\theta}\log f(x_i\mid\theta)}_{Z}\;\underbrace{\prod f(x_i\mid\theta)}_{pdf}\;dx_n\cdots dx_1
$$

---
### Simplifying $Z \times pdf$

Expanding the sum and using $\dfrac{\partial}{\partial\theta}\log f = \dfrac{1}{f}\dfrac{\partial f}{\partial\theta}$, the $\dfrac{1}{f(x_i\mid\theta)}$ cancels with the matching factor in the product:

$$
\begin{aligned}
\sum\frac{\partial}{\partial\theta}\log f(x_i\mid\theta)\cdot\prod f(x_i\mid\theta)
&= \left[\frac{\partial}{\partial\theta}\log f(x_1\mid\theta) + \frac{\partial}{\partial\theta}\log f(x_2\mid\theta) + \cdots\right]\cdot f(x_1\mid\theta)f(x_2\mid\theta)\cdots \\[8pt]
&= \left[\frac{1}{f(x_1\mid\theta)}\frac{\partial}{\partial\theta}f(x_1\mid\theta) + \frac{1}{f(x_2\mid\theta)}\frac{\partial}{\partial\theta}f(x_2\mid\theta) + \cdots\right]\cdot f(x_1\mid\theta)f(x_2\mid\theta)\cdots \\[8pt]
&= \frac{\partial}{\partial\theta}f(x_1\mid\theta)\cdot f(x_2\mid\theta)\cdots f(x_n\mid\theta) \\
&\quad + f(x_1\mid\theta)\cdot\frac{\partial}{\partial\theta}f(x_2\mid\theta)\cdots f(x_n\mid\theta) \\
&\quad\;\;\vdots \\
&\quad + f(x_1\mid\theta)\cdot f(x_2\mid\theta)\cdots\frac{\partial}{\partial\theta}f(x_n\mid\theta) \\[8pt]
&= \frac{\partial}{\partial\theta}\left[f(x_1\mid\theta)\cdot f(x_2\mid\theta)\cdots f(x_n\mid\theta)\right] \\[8pt]
&= \frac{\partial}{\partial\theta}\prod f(x_i\mid\theta)
\end{aligned}
$$

The last two lines are the product rule in reverse. The $n$ terms are exactly the derivative of the product.

---
### Back to $E[TZ]$

Substituting this back in, and swapping $\dfrac{\partial}{\partial\theta}$ and the integrals (regularity),

$$
\begin{aligned}
E[TZ] &= \int_{x_1}\int_{x_2}\cdots\int_{x_n} t(x_1,\ldots,x_n)\,\frac{\partial}{\partial\theta}\prod f(x_i\mid\theta)\;dx_n\cdots dx_1 \\[8pt]
&= \frac{\partial}{\partial\theta}\int_{x_1}\int_{x_2}\cdots\int_{x_n} \underline{t(x_1,\ldots,x_n)}\;\underline{\prod f(x_i\mid\theta)}_{\,pdf}\;dx_n\cdots dx_1 \\[8pt]
&= \frac{\partial}{\partial\theta}E[T] \\[8pt]
&= \frac{\partial}{\partial\theta}\theta = 1
\end{aligned}
$$

So $cov[T,Z]=1$.

---
### Conclusion

Plugging $cov[T,Z]=1$ into $var[T] \ge \dfrac{\{cov[T,Z]\}^2}{nI(\theta_0)}$ gives

$$
\boxed{var[T] \ge \frac{1}{nI(\theta_0)}} \qquad \blacksquare
$$

---
## Comments
- [[Cramer-Rao Inequality|Cramer-Rao Inequality]] gives us lower bound of variance for all the [[Estimator|unbiased estimators]].
- If you have several unbiased estimators, none of them will have a variance lower than $\frac{1}{nI(\theta_{0})}$.
- So if we can find an unbiased estimator whose variance is $\frac{1}{nI(\theta_{0})}$, we know that we have an efficient one.

Note that we showed for large $n$, [[MLE Distribution|MLE]] is unbiased and has a variance of 
$$
\frac{1}{nI(\theta_{0})} \implies \text{MLE is asymptotically efficient}
$$

---
## Examples on Poisson Distribution
For calculating [[Cramer-Rao Inequality|CRLB]] of $\text{Poisson}(\lambda)$, we can do the following.

**Step 1:** log-likelihood
$$
l(\lambda) = -n\lambda + \sum_{i=1}^n X_i \ln\lambda + const.
$$

**Step 2:** Score
$$
\frac{\partial l(\lambda)}{\partial\lambda} = -n + \sum_{i=1}^n \frac{X_i}{\lambda}
$$

**Step 3:** Second derivative
$$
\frac{\partial^2 l(\lambda)}{\partial\lambda^2} = -\sum_{i=1}^n \frac{X_i}{\lambda^2}
$$

**Step 4:** Fisher Information
$$
-E\left[\frac{\partial^2 l(\lambda)}{\partial\lambda^2}\right] = -E\left[-\sum_{i=1}^n \frac{X_i}{\lambda^2}\right] = \frac{1}{\lambda^2}E\left[\sum_{i=1}^n X_i\right] = \boxed{\frac{n}{\lambda}}
$$

**Step 5:** CRLB. Inverting the quantity from step 4, we get
$$
\boxed{CRLB = \frac{\lambda}{n}}
$$

---
### Notes
Note the following
- MLE of $\lambda$ is $\bar{X}$ and $var[\bar{X}] = \frac{\lambda}{n}$
- Therefore, $\bar{X}$ is the efficient estimator out of all unbiased estimators.

---
