# Equality of Two Variances
Suppose we have two independent [[Gaussian Distribution (Normal Distribution)|normal distributions]]:
- $X_{1}, X_{2}, \dots, X_{n} \sim N(\mu_{x},\ \sigma^{2}_{x})$
- $Y_{1}, Y_{2}, \dots, Y_{n} \sim N(\mu_{y},\ \sigma^{2}_{y})$

We want to test the following:
$$
H_{0}: \sigma^{2}_{x} = \sigma^{2}_{y}  \quad \text{v.s} 
\quad H_{1}: \sigma^{2}_{x} \neq \sigma^{2}_{y}
$$
We can then write that
$$
\frac{(n-1)S^{2}_{x}}{\sigma^{2}_{x}} \sim \chi^{2}_{(n-1)}
\quad , \quad
\frac{(m-1)S^{2}_{y}}{\sigma^{2}_{y}} \sim \chi^{2}_{(m-1)}
$$
We can then use that
$$
\begin{align}
&\frac{\frac{(n-1)S^{2}_{x}}{\sigma^{2}_{x}} / (n-1)} 
{\frac{(m-1)S^{2}_{y}}{\sigma^{2}_{y}} / (m-1)} \sim F_{n-1, \ m-1} \\[6pt]

\implies & \frac{S^{2}_{x}/\sigma^{2}_{x}}{S^{2}_{y}/\sigma^{2}_{y}} 
\sim F_{n-1, \ m-1} \\[6pt]

\implies & \underbrace{\frac{S^{2}_{x}}{S^{2}_{y}}}_{\text{test stats}}
\sim \underbrace{F_{n-1, \ m-1}}_{\text{dist}} 
\quad \text{by (1)}
\end{align}
$$
where 
- $(1):$ Under $H_{0}$, $\frac{\sigma^{2}_{x}}{\sigma^{2}_{y}} = 1$

> Hence we get that under $H_{0},$ $\frac{S^{2}_{x}}{S^{2}_{y}} \sim F_{n-1, \ m-1}$.

---
## Confidence Interval
Constructing a $\alpha \text{-level}$ confidence interval for $\frac{\sigma^{2}_{y}}{\sigma^{2}_{x}}$, we get
$$
\begin{align}
& F_{\frac{1-\alpha}{2}, \ n-1, \ m-1} < \frac{S^{2}_{x}/\sigma^{2}_{x}} 
{S^{2}_{y}/\sigma^{2}_{y}} < F_{\frac{1+\alpha}{2}, \ n-1, \ m-1} \\[6pt]

\implies &F_{\frac{1-\alpha }{2}, \ n-1, \ m-1} <  
\frac{S^{2}_{x}}{S^{2}_{y}} \cdot \frac{\sigma^{2}_{y}}{\sigma^{2}_{x}}
< F_{\frac{1+\alpha}{2}, \ n-1, \ m-1} \\[6pt]

\implies & \frac{F_{\frac{1-\alpha }{2}, \ n-1, \ m-1}}{S^{2}_{x}/S^{2}_{y}} <  
\frac{\sigma^{2}_{y}}{\sigma^{2}_{x}}
< \frac{F_{\frac{1+\alpha}{2}, \ n-1, \ m-1}}{S^{2}_{x}/S^{2}_{y}}
\end{align}
$$

---
## Rejection Region
This gives us the rejection area of 
$$
\left(-\infty,\, F_{\alpha/2,(n-1,\,m-1)}\right) \cup \left(F_{1-\alpha/2,(n-1,\,m-1)},\, \infty\right)
$$
We reject $H_{0}$ if $\frac{S^{2}_{x}}{S^{2}_{y}}$ falls in the rejection area.

---
## See Also 
- [[STAB57 Week-6]]
- [[Equality of Two Variances]]
- [[Equality of Two Means (Known Variance)]]
- [[Equality of Two Means (Unknown Variances)]]
- [[Comparing Two Population Means (Paired Data)]]

