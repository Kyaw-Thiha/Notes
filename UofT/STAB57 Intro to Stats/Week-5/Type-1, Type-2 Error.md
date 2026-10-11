# Error Types
![image|400](https://notes-media.kthiha.com/Type-1,-Type-2-Error/cc2824e330694ff31781cecdbd8b9ba6.png)

- $P[\text{Type-1 Error}] = \alpha = P[\text{Reject } H_{0} \mid H_{0} \text{ true}]$
- $P[\text{Type-2 Error}] = \beta = P[\text{Fail to reject } H_{0} \mid H_{0} \text{ false}]$
- $\text{Power of a Test} = 1 - \beta = P[\text{Reject } H_{0} \mid H_{0} \text{ false}]$

---
## Type-1 and Type-2 Error using Graphs
![image|400](https://notes-media.kthiha.com/Type-1,-Type-2-Error/dfa8768ad9e2d51048a23ab1c36a7526.png)

> For a given sample size, decreasing one type will increase the other.

---
## Power of a Test using Graphs
![image|400](https://notes-media.kthiha.com/Type-1,-Type-2-Error/930136e451ac932fad44390f12ecd2ee.png)

---
## Example
Suppose we have $N(\mu, \sigma^{2})$ populations with unknown $\mu$ and $\sigma=3$. We want to test $H_{0}: \mu=1$ $v.s$ $H_{1}: \mu=4$ at $\alpha=0.05$.  We decide to take $n=9$ observations.
Calculate the $P[\text{Type-2 error}]$ and the power.

### Solution
First, we compute the variance to get
$$
var[\bar{X}] = \frac{\sigma^{2}}{n}
= \frac{3^{2}}{9} = 1
$$
Under the [[Hypothesis Test|hypothesis]] $H_{0}$, we get $\bar{X} \sim N(1, 1)$.
Under the [[Hypothesis Test|hypothesis]] $H_{1}$, we get $\bar{X} \sim N(4,1)$.

Recall that $\frac{\bar{X}-\mu}{\sigma/\sqrt{ n }} \sim N(0,1)$. We reject $H_{0}$ if 
$$
\begin{align}
&\frac{\bar{X} - \mu}{\sigma/\sqrt{ n }} > z_{0.95} \mid_{\mu=1} \\[6pt]
\implies &\bar{X} - \mu > z_{0.95} \frac{\sigma}{\sqrt{ n }} \mid_{\mu=1}   
\\[6pt]
\implies &\bar{X} > \mu + z_{0.95} \frac{\sigma}{\sqrt{ n }} \mid_{\mu=1}   
\\[6pt]
\implies &\bar{X} > 1 + 1.645 \frac{3}{\sqrt{ 9 }} \mid_{\mu=1}   
\\[6pt]
\implies &\bar{X} > 2.645 \\[6pt]
\end{align}
$$

Computing the power, we can get
$$
\begin{align}
\text{power}
&= P[ \ \text{rejecting } H_{0} \mid false \ ] \\[6pt]
&= P[ \ \bar{X} > 2.645 \mid \mu=4 \ ] \\[6pt]
&= P\left[  \  \frac{\bar{X} - \mu}{\sigma/\sqrt{ n }}  
> \frac{2.645 - 4}{3/\sqrt{ n }} \  \right] \\[6pt]
&= P[ \ z > -1.355 \ ] \\[6pt]
&= 1 - \text{pnorm}(-1.355) \\[6pt]
&= 0.912
\end{align}
$$

Hence computing the [[Type-1, Type-2 Error|type-2 error]], we get
$$
P[\text{Type-II Error}] = 1 - 0.912 = 0.088
$$

---
