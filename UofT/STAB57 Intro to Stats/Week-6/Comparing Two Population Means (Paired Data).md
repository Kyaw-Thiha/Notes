# Paired Data
In many practical settings, the samples could be paired. 

For examples, if we want to test whether a new drink changes blood sugar level or not, we would measure the blood sugar level of participants before and after. Since the two set of measurements come from same set of individuals, they are not independent.

---
## Solving Paired Data
Suppose the following:
- Let $X$ represent the measurement before the drink.
- Let $Y$ represent the measurement after the drink.

We then want to test the following: 
$$
H_{0}: \mu_{x} - \mu_{y} = 0
\quad \quad \text{vs} \quad \quad
H_{1}: \mu_{x} - \mu_{y} \neq 0
$$
We could still use $\bar{X} - \bar{Y}$ as done in [[Equality of Two Means (Known Variance)|here]] and [[Equality of Two Means (Unknown Variances)|here]], but $var(\bar{X} - \bar{Y})$ will contain a covariance term.

To simplify this problem, we could define the following
$$
D = X-Y
\implies \mu_{d} = \mu_{x} + \mu_{y}
$$
This means $H_{0}:\mu_{x} - \mu_{y} = 0$ is the same as testing $H_{0}:\mu_{d} = 0$.

We can then use
$$
\frac{\bar{D}}{S_{d}/\sqrt{ n }} \sim t_{(n-1)}
$$
Now the problem is like one sample t-test.

---
## Numerical Example
 Let $X$ and $Y$ represent the before and after measurements of $10$ participants. 
Check whether the drink changes the blood sugar level or not.

$$
\begin{array}{c|cccccccccc}
x & 10.19 & 7.92 & 6.67 & 12.22 & 8.21 & 8.26 & 13.06 & 8.20 & 9.83 & 5.94 \\
y & 7.00 & 7.53 & 6.45 & 1.31 & 5.42 & 2.81 & 6.60 & 0.55 & 3.13 & 5.00 \\
d & 3.19 & 0.39 & 0.22 & 10.91 & 2.79 & 5.45 & 6.46 & 7.65 & 6.70 & 0.94
\end{array}
$$

We have $\bar{d} = 4.47$ and $s_{d} = 3.545106$.
Computing the [[Test Statistics|test statistics]], we get
$$
T = \frac{4.47}{3.545106/\sqrt{ 10 }} = 3.987294
$$
This gets us
$$
t_{0.975, \ df=9} = 2.262
$$
This gets us a rejection region of
$$
(-\infty, - 2.262) \cup (2.262, \infty)
$$
Hence, we reject the hypothesis $H_{0}$. This implies that the drink changes the blood sugar level.

---
## See Also 
- [[STAB57 Week-6]]
- [[Equality of Two Variances]]
- [[Equality of Two Means (Known Variance)]]
- [[Equality of Two Means (Unknown Variances)]]
- [[Comparing Two Population Means (Paired Data)]]
