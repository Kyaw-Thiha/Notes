# Week-10
## Question
Suppose $X_{1}, X_{2}, \dots, X_{n}$ are independently drawn from a Gamma distribution with a pdf
$$
f(x) = \frac{x}{\theta^{2}} e^{-x/\theta}
\ , \qquad x > 0, \quad \theta > 0
$$
with $\mathbb{E}[X_{i}] = 2\theta$.

$(a)$ Show that the MLE of $\theta$ is $\bar{X}/2$.
$(b)$ Calculate the Fisher information for the entire sample.
$(c)$ Using the score and Fisher information, construct an $\alpha$-level Score test for $H_{0}: \theta = \theta_{0}$ versus $H_{0}: \theta \neq \theta_{0}$. Give the test statistics, the associated distribution, and the rejection region.

You can use the following:
$$
z_{0.025} = -1.96 \qquad z_{0.975} = 1.96
$$

---
### Question-(a)
> $(a)$ Show that the MLE of $\theta$ is $\bar{X}/2$.

Taking the log likelihood, we get
$$
L(\theta) = \sum ^{n}_{i=1} \log X_{i} - 2n \log \theta - \frac{\sum X_{i}}{\theta}
$$
Taking the first derivative, we get
$$
L'(\theta) 
= -\frac{2n}{\theta} + \frac{\sum X_{i}}{\theta^{2}} 
= \frac{n(\bar{X} - 2\theta)}{\theta^{2}}
$$
Setting it to zero, we get
$$
L'(\theta) = 0 
\implies \frac{n(\bar{X} - 2\theta)}{\theta^{2}} = 0
\implies \hat{\theta}_{MLE} = \frac{\bar{X}}{2}
$$
Taking the second derivative, we get
$$
L''(\theta) = \frac{2n}{\theta^{2}} - \frac{2n \bar{X}}{\theta^{3}}
$$
At $\hat{\theta}$, we get
$$
L''(\hat{\theta}) 
= \frac{2n}{\hat{\theta}^{2}} - \frac{2n\bar{X}}{\hat{\theta}^{3}}
= \frac{2n}{\hat{\theta}^{2}} - \frac{2}{\bar{X}} \cdot \frac{2n\bar{X}}{\hat{\theta}^{2}}
= \frac{2n}{\hat{\theta}^{2}} - \frac{4n}{\hat{\theta}^{2}}
= \frac{2n}{\hat{\theta}^{2}}
$$

---
### Question-(b)
> $(b)$ Calculate the Fisher information for the entire sample.

First note that
$$
\begin{align}
E(\bar{X})
&= E\left( \frac{1}{n} \sum X_{i} \right) \\[6pt]
&= \frac{1}{n} \sum E(X_{i}) \\[6pt]
&= \frac{1}{n} \cdot 2\theta \cdot n \\[6pt]
&= 2\theta
\end{align}
$$

Computing the Fisher information, we get
$$
\begin{align}
\text{Fisher Information}
&= nI(\theta) \\[6pt]
&= -E_{\theta}[L''(\theta)] \\[6pt]
&= -E\left( \frac{2n}{\theta^{2}} - \frac{2n \bar{X}}{\theta^{3}} \right) \\[6pt]
&= -\left( \frac{2n}{\theta^{2}} - \frac{2n \cdot E(\bar{X})}{\theta^{3}} \right)
\\[6pt]
&= \frac{2n}{\theta^{2}}
\end{align}
$$

---
### Question-(c)
> $(c)$ Using the score and Fisher information, construct an $\alpha$-level Score test for $H_{0}: \theta = \theta_{0}$ versus $H_{0}: \theta \neq \theta_{0}$. Give the test statistics, the associated distribution, and the rejection region.

Recall that 
$$
Z = \frac{S(\theta_{0})}{\sqrt{ nI(\theta_{0}) }}
$$
where $S(\theta_{0})$ is the $\text{score} = l'(\theta)$ and $nI(\theta_{0})$ is the Fisher information.

Computing it, we get
$$
Z = \frac{\frac{n(\bar{X} - 2\theta_{0})}{\theta^{2}_{0}}}{\sqrt{ \frac{2n}{\theta^{2}_{0}} }}
= \frac{\sqrt{ n } (\bar{X} - 2\theta_{0})}{\sqrt{ 2 } \theta_{0}}
$$
$\therefore$ This is 2-tail. Hence, the rejection region is 
$$
\left( -\infty, \ Z_{\frac{\alpha}{2}} \right) \cup \left( Z_{1 - \frac{\alpha}{2}}, \ \infty \right)
$$

---