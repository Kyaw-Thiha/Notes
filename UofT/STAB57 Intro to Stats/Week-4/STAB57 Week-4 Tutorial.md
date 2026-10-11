# Week-4 Tutorial
## Question-1
Suppose we want to construct a [[Confidence Interval|confidence interval]] for average$(\mu)$ daily screen time(hour) of university students. We can assume that daily screen time is [[Gaussian Distribution (Normal Distribution)|normally distributed]].

We observe the following daily screen time:
$$
5.2 \quad 6.8 \quad 7.1 \quad 4.9 \quad 6.3 \quad 7.5 \quad 5.8 \quad 6.1 \quad 8.0 \quad 5.5
$$
Construct an $80\%$ confidence interval for $\mu$ and interpret it.

### Solution

If $\sigma$ is known $\implies$ $\bar{x} \pm z_{\frac{\alpha}{2}} \frac{\sigma}{\sqrt{ n }}$
If $\sigma$ is unknown $\implies$ $\bar{x} \pm z_{\frac{\alpha}{2}, n-1} \frac{s}{\sqrt{ n }}$

Computing the mean, we get
$$
\bar{x} = \frac{5.2 + \dots + 5.5}{10}
= 6.32
$$
Next, we compute the unbiased variance to get
$$
s = \sqrt{ \frac{1}{n-1} \sum(x_{i} - \bar{x})^{2} }
= \sqrt{ \frac{1}{n-1} ( \sum(x_{i}^{2}) - n\cdot(\bar{x})^{2} })
= \sqrt{ \frac{9.316}{9} }
$$

Since $\gamma=0.8$, we have $\frac{1+0.8}{2} = 0.9$. Using t-table, we then get
$$
t_{0.9, \ 9} \approx 1.383
$$
Hence, the $0.8$-CI for $\mu$ is 
$$
\left(
6.32 - 1.383 \left( \frac{1.017}{\sqrt{ 10 }} \right), \ 
6.32 + 1.383 \left( \frac{1.017}{\sqrt{ 10 }} \right)
\right)
= (5.875, 6.765)
$$

> We are $80\%$ confident that the average daily screen time of university students is between $5.875$ and $6.765$ hours.

---
## Question-2
Suppose we want to construct a $90\%$ [[Confidence Interval|confidence interval]] for the proportion of UTSC students who regularly use the campus gym.

How many students should we select, if we want margin of error to be less than $0.05$?

### Solution
For a population proportion: 
$$
\hat{p} \pm \underbrace{z_{\frac{\alpha}{2}} \sqrt{ \frac{\hat{p}(1-\hat{p})}{n} }}_{\text{Margin of Error}}
$$
We then have
$$
\begin{align}
\text{Margin of Error} &= Z_{\frac{\sigma}{2}} \sqrt{ \frac{p(1-p)}{n} } \\[6pt]
n &= \frac{Z_{\frac{\alpha}{2}}^{2}  \ p(1-p)}{ME^{2}} \\[6pt]
n &= \frac{Z_{\frac{0.1}{2}}^{2}  \ 0.5(1-0.5)}{0.05^{2}} \\[6pt]
n &= \frac{(1.645)^{2} (0.5)(0.5)}{(0.05)^{2}} \\[6pt]
n &= 270.60
\end{align}
$$
Note that we **always round up** for $n$. Hence, we have $n=271$ students.

---