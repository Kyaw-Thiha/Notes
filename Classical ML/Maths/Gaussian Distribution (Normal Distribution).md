# Gaussian Distribution (Normal Distribution)
A [[Gaussian Distribution (Normal Distribution)|normal distribution]] is a distribution that is symmetric around the mean, and forming the bell-shaped curve.

![Gaussian Distribution|300](https://articles.outlier.org/_next/image?url=https%3A%2F%2Fimages.ctfassets.net%2Fkj4bmrik9d6o%2FUuFO0KrA3JNEZfBsG8bSc%2F9fd6ec46281d7e907820cdd1cb75bff9%2FNormal_Distribution_07.png&w=3840&q=75)


---
## PDF
$$
f_X(x) = \frac{1}{\sqrt{2\pi \alpha^2}} 
\exp\!\left( -\frac{(x - \mu)^2}{2\alpha^2} \right),
\quad x \in \mathbb{R},
$$

where
- $\mu = \mathbb{E}[X] \in \mathbb{R}, \quad$
- $\alpha^2 = \mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2] \in \mathbb{R}.$

---
## Standard Normal Distribution ($Z$-Distribution)
If $X \sim \mathbb{N}(\mu, \sigma^{2})$, then $Z = \frac{x - \mu}{\sigma} \sim \mathbb{N}(0,1)$.
This [[#Z-Distribution]] has 
- $\mathbb{E}[Z] = 0$
- $V[Z] = 1$
- This implies that $\mathbb{E}[X^{2}] = 1$.

---
## $\mathcal{X}^{2}$ Distribution
Let $U = Z^{2}$, where $Z$ is a [[#Z-Distribution|Standard Normal Distribution]].
Then, $U \sim \mathcal{X}_{(1)}^{2}$ is a distribution with $1$ degrees of freedom.

**Additive Property**: Suppose $X \sim \mathcal{X}^{2}_{(m)}$ and $Y \sim \mathcal{X}^{2}_{(n)}$, and $X$ and $Y$ are independent. 
Then,
$$
X + Y \sim \mathcal{X}^{2}_{(m+n)}
$$

**Expectation**: If $X \sim \mathcal{X}^{2}_{(m)}$, then $\mathbb{E}[X] = m$.
$$
\begin{align}
\mathbb{E}[\mathcal{X}^{2}_{(m)}]
&= \mathbb{E}[ \ \mathcal{X}_{(1)}^{2} + \mathcal{X}_{(1)}^{2} + \dots 
+ \mathcal{X}_{(1)}^{2} \ ] \\[6pt]
&= \mathbb{E}[z_{1}^{2} + z_{2}^{2} + \dots + z_{m}^{2}] \\[6pt]
&= \mathbb{E}[z_{1}^{2}] + \mathbb{E}[z_{2}^{2}] + \dots  
+ \mathbb{E}[z_{m}^{2}] \\[6pt]
&= 1 + 1 + \dots + 1 \\[6pt]
&= m
\end{align}
$$

---
## $t$-Distribution
Let $Z \sim \mathbb{N}(0,1)$ and $U \sim \mathcal{X}^{2}_{(m)}$ be two independent variables.
Then,
$$
\frac{Z}{\sqrt{ U/m }} \sim t_{(m)}
$$
is a [[#$t$-Distribution|t-distribution]] with $m$ degrees of freedom.

Let $T = \frac{z}{\sqrt{ U/m }} \sim t_{(m)}$. This implies that
$$
T^{2} = \frac{z^{2} / 1}{u/m} \sim F_{(1,m)}
$$
is a [[#$F$ Distribution|F-distribution]].

---
## $F$ Distribution
Let $X \sim \mathcal{X}^{2}_{(m)}$ and $Y \sim \mathcal{X}^{2}_{(n)}$ be two independent variables.
Then, 
$$
\frac{X/m}{Y/n} \sim F_{(m,n)}
$$
is a [[#$F$ Distribution|F-distribution]] with degrees of freedom $(m,n)$.

Let $w = \frac{x/m}{y/n} \sim \text{F}_{(m,n)}$ . This implies that
$$
\frac{1}{w} \sim F_{(n,m)}
$$

---
