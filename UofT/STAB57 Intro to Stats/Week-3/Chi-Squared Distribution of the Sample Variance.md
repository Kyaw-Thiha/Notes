# Chi-Squared Distribution of the Sample Variance
The [[Chi-Squared Distribution of the Sample Variance]] is defined as
$$
\boxed{ \ \frac{(n-1)S^{2}}{\sigma^{2}} \sim \mathcal{X}^{2}_{(df=n-1)} \ }
$$
where the [[Sample Variance|standardized sample variance]] has the [[Chi-Squared Distribution|chi-squared distribution]].

---
## Proof
### Preliminary
Recall that
$$
\begin{align}
S^{2} &= \frac{1}{n-1} \sum(X_{i} - \bar{X})^{2} \\[6pt]
\implies \sum(X_{i} - \bar{X})^{2} &= (n-1)S^{2}
\end{align}
$$

---
## Decomposing the Total Variation
Using the above, we have
$$
\begin{align}
\sum(X_{i} - \mu)^{2} &= \sum(X_{i} - \bar{X})^{2} + n(\bar{X} - \mu)^{2}  
\\[6pt]
\implies \sum(X_{i} - \mu)^{2} &= (n-1) \ S^{2} + n(\bar{X} - \mu)^{2} \\[6pt]
\end{align}
$$
### Standardizing the Decomposition
Dividing by $\sigma^{2}$, we get
$$
\begin{align}
\frac{\sum(X_{i} - \mu)^{2}}{\sigma^{2}}
&= \frac{(n-1)S^{2}}{\sigma^{2}} + \frac{n(\bar{X} - \mu)^{2}}{\sigma^{2}}
\\[6pt]
\implies \sum \frac{(X_{i} - \mu)^{2}}{\sigma^{2}}
&= \frac{(n-1)S^{2}}{\sigma^{2}} + \frac{(\bar{X} - \mu)^{2}}{\sigma^{2}/n}
\\[6pt]
\implies \sum \left( \frac{X_{i} - \mu}{\sigma} \right)^{2}
&= \frac{(n-1)S^{2}}{\sigma^{2}} +  
\left( \frac{\bar{X} - \mu}{\sigma/\sqrt{ n }} \right)^{2}
\\[6pt]
\end{align}
$$

---
## Distributions of the Two Components
### Distribution of the Total Variation about μ
For $X_{i} \sim \mathbb{N}(\mu, \sigma^{2})$,  we have
$$
\begin{align}
X_{i} &\sim \mathbb{N}(\mu, \sigma^{2}) \\[6pt]
\frac{X_{i} - \mu}{\sigma} &\sim \mathbb{N}(0,1) \\[6pt]
\left( \frac{X_{i} - \mu}{\sigma} \right)^{2} &\sim \mathcal{X}^{2}_{(1)}  
\\[6pt]
\sum^n_{i=1} \left( \frac{X_{i} - \mu}{\sigma} \right)^{2} &\sim \mathcal{X}^{2}_{(n)}  
\\[6pt]
\end{align}
$$

---
### Distribution of the Sample Mean Term
For $\bar{X} \sim \mathbb{N}\left( \mu, \frac{\sigma^{2}}{n} \right)$, we have
$$
\begin{align}
\bar{X} &\sim \mathbb{N}\left( \mu, \frac{\sigma^{2}}{n} \right) \\[6pt]
\frac{\bar{X}-\mu}{\sigma/\sqrt{ n }} &\sim \mathbb{N}\left( 0,1 \right) \\[6pt]
\left( \frac{\bar{X}-\mu}{\sigma/\sqrt{ n }} \right)^{2}  
&\sim \chi^{2}_{(1)} \\[6pt]
\end{align}
$$

---
## Moment Generating Function
Applying the [[Moment Generating Function(MGF)|MGF]] to the earlier equation, we get
$$
\begin{align}
&\sum \left( \frac{X_{i} - \mu}{\sigma} \right)^{2} =
\frac{(n-1)S^{2}}{\sigma^{2}} +  
\left( \frac{\bar{X} - \mu}{\sigma/\sqrt{ n }} \right)^{2}
\\[6pt]

\implies &\text{MGF of } \mathcal{X}^{2}_{(df=n)}
= \left[ \text{MGF of } \frac{(n-1)S^{2}}{\sigma^{2}} \right] 
* [\text{MGF of } \mathcal{X}^{2}_{(df=1)}] \\[6pt]

\implies &(1-2t)^{-n/2}
= \left[ \text{MGF of } \frac{(n-1)S^{2}}{\sigma^{2}} \right] 
* (1-2t)^{-1/2} \\[6pt]

\implies &\left[ \text{MGF of } \frac{(n-1)S^{2}}{\sigma^{2}} \right] 
= (1-2t)^{-(n-1)/2}
\end{align}
$$
which is the [[Moment Generating Function(MGF)|MGF]] of a $\mathcal{X}^{2}_{df=(n-1)}$.

Hence, we get that
$$
\frac{(n-1)S^{2}}{\sigma^{2}} \sim \mathcal{X}^{2}_{(df=n-1)}
\quad \blacksquare
$$

---