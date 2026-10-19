# Week-5 Tutorial
## Question
Suppose the following samples are taken from a [[Gaussian Distribution (Normal Distribution)|Normal distribution]] with known mean $\mu$ and variance $\sigma^{2}$. 
We want to test $H_{0}: \mu = 11$ versus $H_{1}: \mu < 11$ at level of significance $\alpha = 0.05$.

The observations are
$$
8.7, \ 10.9, \ 11.8, \ 9.6, \ 10.2, \ 
12.0, \ 9.4, \ 11.1, \ 10.6, \ 9.7 \ 
$$
For these numbers, $\sum x_{i} = 104.0$ and $\sum(x_{i} - \bar{x})^{2} = 10.36$.

---
## Solutions
- Unknown mean and variance
- $\sum x_{i} = 104.0$ and $\sum(x_{i} - \bar{x}) = 10.36$
- Observation: 

Look at $H_{A}$:
- $H_{A}: \mu \neq \mu_{0} \implies$ two-sided. Critical Values: $\pm t_{1-\frac{\alpha}{2}, df}$
- $H_{A}: \mu < \mu_{0} \implies$ left-tailed. Critical Values: $- t_{1-\alpha, df}$
- $H_{A}: \mu > \mu_{0} \implies$ right-tailed. Critical Values: $+ t_{1-\alpha, df}$

---
### Question (a)
$(a)$ Determine the rejection region and show it on a plot.
Critical Value: $-t_{1-0.05, df}= -t_{1 - 0.05, \ 10-1} = -1.833$
Rejection Region: $(\infty, \ -1.833)$

---
### Question (b)
$(b)$ Calculate the test statistics associated with the test.

Recall that we have
- $\bar{X} = \frac{\sum X_{i}}{n} = \frac{104}{10} = 10.4$
- $\mu_{0} = 11$
- $S^{2} =\frac{\sum(x_{i} - \bar{x})^{2}}{n-1} = \frac{10.36}{10-1} = 1.1511$
$$
\begin{align}
T &= \frac{\bar{X} - \mu_{0}}{S/\sqrt{ n }} \\[6pt]
&= \frac{10.4 - 11}{ \sqrt{ 1.1511 } / \sqrt{ 10 }} \\[6pt]
&= -1.768
\end{align}
$$

---
### Question (c)
$(c)$ What conclusion do you reach? (Whether you reject $H_{0}$ or not)

$$
\begin{align}
-1.768 &> -1.833 \\[6pt]
\implies -1.768 &\not\in (-\infty, \ -1.833)
\end{align}
$$

$\therefore$ We fail to reject $H_{0}$.

---
### Question (d)
Let $G$ represent the CDF of the distribution that you used in part $(a)$. Write the [[P-Value|p-value]] associated with the test in terms of $G$.

$G(-1.768)$ with $df = 9$.

---