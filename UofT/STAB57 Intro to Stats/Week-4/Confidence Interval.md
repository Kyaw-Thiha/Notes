# Confidence Interval
A [[Confidence Interval|confidence interval]] is a range of values that is likely to contain the true value of an unknown population parameter.

![250](https://www.appinio.com/hs-fs/hubfs/Confidence%20Interval%20Appinio.png?width=1024&height=768&name=Confidence%20Interval%20Appinio.png)

---
## Formal Definition
An [[Confidence Interval|interval]] of
$$
C(X_{1}, X_{2}, \dots, X_{n}) = (l(X_{1}, X_{2}, \dots, X_{n}), \ u(X_{1}, X_{2}, \dots, X_{n}))
$$
is a $\gamma \text{-confidence interval}$ for $\psi(\theta)$ if 
$$
\begin{align}
P_{\theta}[\psi(\theta) \in C(X_{1}, X_{2}, \dots, X_{n})] &\geq \gamma \\[6pt]
\implies P_{\theta}[ \ l(X_{1}, X_{2}, \dots, X_{n})] \leq
\psi(\theta) \leq u(X_{1}, X_{2}, \dots, X_{n}) \ ] &\geq \gamma
\end{align}
$$
for every $\theta \in \Omega$.

There, $\gamma$ represents the [[Confidence Interval|confidence level]] of the interval.

---
## Example
Assume the unknown parameter is $\mu$. Assume $\gamma = 0.95$.
Then, we will want an expression similar to 
$$
P[l() \leq \mu \leq u()] \geq 0.95
$$

![image|300](https://notes-media.kthiha.com/Confidence-Interval/5f25c60e85a03c25168658099601df5f.png)

The tool that allows us to compute this is called [[Pivotal Quanity|pivotal quantity]].

---
