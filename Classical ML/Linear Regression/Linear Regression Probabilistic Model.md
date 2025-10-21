# Linear Regression Probabilistic Model

Assume 
$$
\begin{align}
y &= f(x) + \eta \\[6pt]
&= w^T.x + \eta \\
\end{align}
$$

where
- $w \in R^D$
- $x \in R^D$
- $y \in R$
- $\eta \sim N(0, \alpha^2)$

Hence, $E[y] = E[\eta]$ and $Var[y] = Var[\eta]$

$$
P(y|w, x) = \frac{1}{\sqrt{ 2\pi\sigma }}.\exp\left( -\frac{1}{2\sigma^2 } (y - w^T.x)^2 \right)
$$
Since $w^Tx$ is constant and $\eta \sim N(0, \alpha^2)$, $y$ is also a [[Gaussian Distribution]].
Hence, we can find its `MAP` [[Estimation]] by using `Negative Log Posterior` ([[Log Likelihood]]).

Read more at [[MAP for Non-Linear Regression]].
