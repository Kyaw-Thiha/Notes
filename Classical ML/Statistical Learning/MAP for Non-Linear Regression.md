# MAP for Non-Linear Regression

The [[Non-Linear Regression]] model can be defined as 
$$
y = w^Tb(x) + \eta
$$
where $\eta \sim N(0, \sigma^2)$  is the noise

Since $w^Tb(x)$ is deterministic, it is scalar constant.
Hence since noise $\eta \sim N(0, \sigma^2)$, sum of constant + gaussian is also a gaussian.
$$
\begin{align}
p(y | x, w)  
&= G(y; w^Tb(x), \sigma^2) \\[6pt]
&= -\frac{1}{2\pi\sigma} e^{-(y - w^Tb(x))^2 / 2\sigma^2}
\end{align}
$$

For a collection of $N$ independent training data points $(y_{1:N}, x_{1:N})$, 
$$
\begin{align}
p(y_{1:N} | w, x_{1:N})
&= \Pi^N_{i=1} G(y_{i}; \ w^Tb(x_{i}), \sigma^2) \\[6pt]
&= \frac{1}{(2\pi \sigma^2)^{N/2}} \exp\left( - \sum^N_{i=1} \frac{(y_{i} - w^Tb(x_{i}))^2}{2\sigma^2} \right)
\end{align}
$$

Note that for $w \in R^d$,
$$
p(w) = 
\Pi^d_{k=1} \frac{1}{\sqrt{ 2\pi\alpha }} e^{-w_{k}^2 / 2\alpha}
= \Pi^d_{k=1} \frac{1}{ (2\pi\alpha)^{d / 2} } e^{-w^Tw / 2\alpha}
$$

To carry out [[Estimation]] on the model parameters, we can apply [[Bayes Rule in ML]] to get
$$
\begin{align}
p(w | y_{1:N}, x_{1:N})
&= \frac{p(y_{1:N} | w, x_{1:N})( p(w | x_{1:N}) )}{p(y_{1:N} | x_{1:N})} \\[6pt]
&= \frac{\Pi_{i} ( p(y_{i} | w, x_{i})) ( p(w) )}{p(y_{1:N} | x_{1:N})}  
&\text{by } p(w|x_{1:N}) = p(w) \\[6pt]
\end{align}
$$

In `MAP Estimation`, we attempt to find parameter $w$ that maximize the posterior
$$
\begin{align}
w^* 
&= argmax_{w} p(w | y_{1:N}, x_{1:N}) \\[6pt]
&= argmin_{w} - \ln p(w | y_{1:N}, x_{1:N}) \\
\end{align}
$$

Hence, the `Negative Log Posterior` ([[Log Likelihood]]) is 
$$
\begin{align}
L(w)
&= -\ln p(w \ | \ y_{1:N}, x_{1:N}) \\[6pt]

&= \left( \sum_{i} \frac{1}{2\alpha^2} (y_{i} - w^Tb(x_{i}) )^2 \right) + \frac{N}{2} \ln(2\pi \sigma^2) + \frac{1}{2\alpha} ||w||^2  + \frac{d}{2} \ln(2\pi\alpha)  \\
&+ \ln p(y_{1:N} | x_{1:N}) \\[6pt]

&= \left( \sum_{i} \frac{1}{2\alpha^2} (y_{i} - w^Tb(x_{i}) )^2 \right) + \frac{1}{2\alpha} ||w||^2 + \text{constants}
& \text{by removing terms without } w \\[6pt]

&= \left( \sum_{i} (y_{i} - w^Tb(x_{i}) )^2 \right) + \frac{\sigma^2 }{\alpha} ||w||^2 + \text{constants}
& \text{multiplying by } 2\sigma^2 \\[6pt]

&= \left( \sum_{i} (y_{i} - w^Tb(x_{i}) )^2 \right) + \lambda ||w||^2 + \text{constants}
& \text{by } \lambda = \frac{\sigma^2}{\alpha} \\[6pt]
\end{align}
$$

Note that $L(w) = \sum_{i} (y_{i} - w^Tb(x_{i}) )^2 + \lambda ||w||^2 + \text{constants}$ is the objective function of [[Non-Linear Regression]].
Hence, non-linear least squares is a form of `MAP Estimation`.

This also show that when measurements are reliable, $\sigma$ is small and `regularizer` have less influence.
But when measurements are noisy, $\sigma$ is larger and `regularizer` has more influence.
