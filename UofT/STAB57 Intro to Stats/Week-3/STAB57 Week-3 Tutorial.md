> This material is taught in Week-4, but covers Week-3 material. 

## Question-1
Suppose $X_{1}, X_{2}, X_{3}, X_{4}$ are independent random variables with $X_{1} \sim N(\mu, \sigma^{2})$.
Consider two estimators of $\mu$:
- $T_{1} = \frac{X_{1} + 2X_{2} + X_{3}}{3}$
- $T_{2} = \frac{2X_{1} - X_{2} + X_{3} + 2X_{4}}{4}$

By calculating the [[Bias-Variance|bias]], [[Bias-Variance|variance]], and [[Mean Squared Error(MSE)|MSE]] suggest which one is more accurate.

### Answer
Recall the following
$$
\begin{align}
\text{Bias}(T) &= \mathbb{E}(T) - \theta \\[6pt]
\text{MSE}(T) &= Var(T) + (\text{Bias}(T))^{2}
\end{align}
$$
#### Estimator-1 $(T_{1})$
Calculating the expectation of $T_{1}$, we get
$$
\begin{align}
E(T_{1}) 
&= E\left( \frac{X_{1} + 2X_{2} + X_{3}}{3} \right) \\[6pt]
&= \frac{1}{3} E(X_{1}) + \frac{2}{3} E(X_{2}) + \frac{1}{3} E(X_{3}) \\[6pt]
&= \frac{1}{3} \mu + \frac{2}{3} \mu + \frac{1}{3} \mu \\[6pt]
&= \frac{4}{3}\mu
\end{align}
$$
Hence, we get the bias of
$$
\text{Bias}(T_{1}) = \frac{4}{3} \mu - \mu = \frac{1}{3} \mu
$$
Next, we calculate the variance to get
$$
\begin{align}
Var(T_{1})
&= Var\left( \frac{X_{1}+ 2X_{2} + X_{3}}{3} \right) \\[6pt]
&= \frac{1}{9} Var(X_{1}) + \frac{4}{9} Var(X_{2}) + \frac{1}{9} Var(X_{3})
\\[6pt]
&= \frac{2}{3} \sigma^{2}
\end{align}
$$

Computing the [[Mean Squared Error(MSE)|MSE]], we get
$$
MSE(T_{1}) = \frac{2}{3} \sigma^{2} + \left( \frac{1}{3} \mu \right)^{2}
= \frac{2}{3}\sigma^{2} + \frac{1}{9}\mu^{2}
$$

---
#### Estimator-2 $(T_{2})$
Calculating the expectation, we get
$$
\begin{align}
E(T_{2})
&= E\left(\frac{2X_{1} - X_{2} + X_{3} + 2X_{4}}{4} \right) \\[6pt]
&= \frac{2}{4} E(X_{1}) - \frac{1}{4} E(X_{2}) + \frac{1}{4} E[X_{3}]
+ \frac{2}{4} E[X_{4}] \\[6pt]
&= \mu
\end{align}
$$
Hence, we get the bias of
$$
Bias(T_{2}) = \mu - \mu = 0
$$

Computing the variance, we get
$$
\begin{align}
Var(T_{2}) &= Var\left( \frac{2X_{1} - X_{2} + X_{3} + 2X_{4}}{4} \right)
\\[6pt]
&= \frac{1}{16} (4\sigma^{2} + \sigma^{2} + \sigma^{2} + 4\sigma^{2}) \\[6pt]
&= \frac{5}{8} \sigma^{2}
\end{align}
$$

Hence, the [[Mean Squared Error(MSE)|MSE]] is
$$
\text{MSE}(T_{2}) = \frac{5}{8} \sigma^{2}
$$

---
#### Comparism
$$
\begin{align}
\frac{2}{3} \sigma^{2} + \frac{1}{9}\mu^{2} &> \frac{5}{8} \sigma^{2} \\[6pt]
T_{1} &\geq T_{2}
\end{align}
$$

Therefore, $T_{2}$ is more accurate.

---
## Question-2
Suppose $X_{1}, X_{2}, \dots, X_{n} \overset{iid}{\sim} N(\mu, \sigma^{2})$, and you are told that $\frac{(n-1)S^{2}}{\sigma^{2}} \sim \mathcal{X}^{2}_{n-1}$, where $S^{2}$ is the [[Sample Variance|sample variance]]. Calculate the variance of
$$
T = \frac{3}{2n-1} \sum^{n}_{i=1} (X_{i} - \bar{X})^{2}
$$
as estimator of $\sigma^{2}$.

Note that
- $Y \sim \chi^{2}_{K}$
- $Var(T) = 2k$

### Solution
Recall that the variance is
$$
\begin{align}
&Var\left( \frac{2}{2n-1} \sum ^{n}_{i=1} (X_{i} - \bar{X})^{2} \right)
\\[6pt]
\end{align}
$$
#### Getting $T$
We can then get
$$
\begin{align}
&S^{2} =  \frac{2}{2n-1} \sum ^{n}_{i=1} (X_{i} - \bar{X})^{2} 
\\[6pt]
\iff &\sum ^{n}_{i=1} (X_{i} - \bar{X})^{2} = (n-1)S^{2} \\[6pt]
\end{align}
$$
Substituting it into $T$, we have
$$
\begin{align}
T &= \frac{3}{2n-1} \sum^{n}_{i=1} (X_{i} - \bar{X})^{2} \\[6pt]
&= \frac{3}{2n-1} \cdot (n-1) \cdot S^{2} \\[6pt]
&= \frac{3(n-1)S^{2}}{2n-1}
\end{align}
$$
---
#### Getting Variance of S
Separately, we have
$$
\begin{align}
&\frac{(n-1)S^{2}}{\sigma^{2}} \sim \chi^{2}_{n-1} \\[6pt]
\implies &\sigma^{2} \cdot \chi^{2}_{n-1} = (n-1)S^{2} \\[6pt]
\implies &\frac{\sigma^{2}}{n-1} \chi^{2}_{n-1} = S^{2}
\end{align}
$$
Putting it back into the sample variance, we get
$$
\begin{align}
Var(S^{2}) &= Var\left( \frac{\sigma^{2}}{n-1} \chi^{2}_{n-1} \right) \\[6pt]
&= \left( \frac{\sigma^{2}}{n-1} \right)^{2} Var(\chi^{2}_{n-1}) \\[6pt]
&= \frac{\sigma^{4}}{(n-1)^{2}} \cdot 2(n-1) \\[6pt]
&= \frac{2\sigma^{4}}{n-1}
\end{align}
$$

---
#### Getting Variance of T
$$
\begin{align}
Var(T)
&= Var\left( \frac{3()}{} \right)
\end{align}
$$