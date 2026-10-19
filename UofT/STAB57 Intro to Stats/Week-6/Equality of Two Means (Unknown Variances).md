# Equality of Two Means (Unknown Variances)
## Equal Variances
> Testing $\mu_{x} = \mu_{y}$ with unknown $\sigma_{x} = \sigma_{y} = \sigma$.

We have that
- $\bar{X} \sim \mathcal{N}\left( \mu_{x}, \ \frac{\sigma^{2}}{n} \right)$  
- $\bar{Y} \sim \mathcal{N}\left( \mu_{y}, \ \frac{\sigma^{2}}{m} \right)$

Therefore, we have that
$$
\frac{\overline{X} - \overline{Y}}{\sigma \ \sqrt{\dfrac{1}{n} 
+ \dfrac{1}{m}}} \sim N(0, 1)
$$

Using the $\mathcal{X}^{2}$ properties, 
$$
\begin{align*}
& \frac{(n-1)S_x^2}{\sigma^2} + \frac{(m-1)S_y^2}{\sigma^2} \sim \chi^2_{(n-1)} + \chi^2_{(m-1)} = \chi^2_{(n+m-2)} \\[1em]
\implies & \frac{1}{\sigma^2}\left[(n-1)S_x^2 + (m-1)S_y^2\right] \sim \chi^2_{(n+m-2)}
\end{align*}
$$

---
## 3 Options
The $df$ of $t$-distribution under $3$ options:
- $t_{(n-1)}$
- $t_{(m-1)}$
- $t_{(n+m-2)}$

---
### Option-1
$$
\frac{(n-1) \ S^{2}_{y}}{\sigma^{2}} \sim \chi^{2}_{(n-1)}
$$

---
### Option-2
$$
\frac{(m-1) \ S^{2}_{y}}{\sigma^{2}} \sim \chi^{2}_{(m-1)}
$$

---
### Option-3
$$
\begin{align*}
& \frac{(n-1)S_x^2}{\sigma^2} + \frac{(m-1)S_y^2}{\sigma^2} \sim \chi^2_{n+m-2} \\[1em]
\implies & \frac{1}{\sigma^2}\left[(n-1)S_x^2 + (m-1)S_y^2\right] \sim \chi^2_{n+m-2}
\end{align*}
$$

---
## Under t-distribution
Using the definition of a $t$-distribution,
$$
\begin{align*}
& \frac{\dfrac{\overline{X} - \overline{Y}}{\sigma\sqrt{\dfrac{1}{n} + \dfrac{1}{m}}}}{\sqrt{\dfrac{\dfrac{1}{\sigma^2}\left[(n-1)S_x^2 + (m-1)S_y^2\right]}{n+m-2}}} \sim t_{n+m-2} \\[1.5em]
\implies & \frac{\overline{X} - \overline{Y}}{\sqrt{\dfrac{(n-1)S_x^2 + (m-1)S_y^2}{n+m-2}}\,\sqrt{\dfrac{1}{n} + \dfrac{1}{m}}} \sim t_{(n+m-2)} \\[1.5em]
\implies & \frac{\overline{X} - \overline{Y}}{S_p\sqrt{\dfrac{1}{n} + \dfrac{1}{m}}} \sim t_{(n+m-2)}
\end{align*}
$$

where $S_p^2 = \frac{(n-1)S_x^2 + (m-1)S_y^2}{n+m-2}$ is called the [[Sample Variance|pooled sample variance]].

---
## Unequal Variances
> Testing $\mu_{x} = \mu_{y}$ with unknown variances and $\sigma_{x} \neq \sigma_{y}$.

We shall use $\frac{S^{2}_{x}}{n} + \frac{S^{2}_{y}}{m}$ as an [[Estimator|estimator]] of $\frac{\sigma^{2}_{x}}{n} + \frac{\sigma^{2}_{y}}{m}$. 
We can then use the test statistics of
$$
T = \frac{\bar{X} - \bar{Y}}{\sqrt{ \frac{S^{2}_{x}}{n} + \frac{S^{2}_{y}}{m} }}
$$

---
## $df$ Calculation
The distribution of the [[Test Statistics|test statistics]] provided above is approximated using a t-distribution with
$$
df = \frac{\left(\dfrac{S_x^2}{n} + \dfrac{S_y^2}{m}\right)^2}{\dfrac{\left(S_x^2/n\right)^2}{n-1} + \dfrac{\left(S_y^2/m\right)^2}{m-1}}
$$

---
## See Also 
- [[STAB57 Week-6]]
- [[Equality of Two Variances]]
- [[Equality of Two Means (Known Variance)]]
- [[Equality of Two Means (Unknown Variances)]]
- [[Comparing Two Population Means (Paired Data)]]
