# Monte Carlo Methods

`[1]` Generate $\{ x_{i} \}^N_{i=1}$ from $P(x)$
`[2]` Approximate $E_{x \sim p(x)} [\ \phi(x) \ ] = \int_{x} \phi(x) p(x) dx$

---
## Simple Monte Carlo

To learn $\{ x_{i} \}^N_{i=1} \sim p(x)$, we can find
$$
E_{x \sim p(x)}[ \ \phi(x) \ ] 
\approx \frac{1}{N} \sum^N_{i=1} \phi(x_{i})
$$

This is also called `Sample Mean Estimator`.
Because, we are getting the mean of the samples.

### Properties
Let $\Phi := E_{x \sim p(x)} [ \ \phi(x) \ ]$ and $\hat{\Phi} := \frac{1}{N} \sum^N_{i=1} \phi(x_{i})$

`[1]` Consistent: $\hat{\Phi} \to \Phi, \text{ as } N \to \infty$

`[2]` Unbiased: $E[ \ \hat{\Phi} \ ] = \Phi$

`[3]` As value of $N$ increases, $Var(\hat{\Phi})$ decreases at the rate of $\frac{1}{N}$

### Properties Proof
`[2] Proof of Expectation`
$$
\begin{align}
E[\hat{\Phi}] 
&= E\left[ \frac{1}{N} \sum^N_{i=1} \phi(x_{i}) \right] \\[6pt]

&= \frac{1}{N} \sum^N_{i=1} E_{x \sim p(x)} \left[  \phi(x_{i}) \right] \\[6pt]

&= E_{x \sim p(x)} [\Phi(x)] \\[6pt]
&= \Phi
\end{align}
$$

`[3] Proof of Variance`
$$
\begin{align}
Var(\hat{\Phi})
&= Var\left( \frac{1}{N} \sum^N_{i=1} \phi(x_{i}) \right) \\[6pt]

&= \frac{1}{N^2} Var\left(  \sum^N_{i=1} \phi(x_{i}) \right) \\[6pt]

&= \frac{1}{N^2} \sum^N_{i=1} Var\left(   \phi(x_{i}) \right) \\[6pt]

&= \frac{1}{N} \ Var(\phi(x)) \\[6pt]
\end{align}
$$

---

## Sampling Strategies (Gaussian)

### 1-Dimensional Sampling

`[1]` $x \sim N(0, 1)$
Use `Box-Muller Method`

`[2]`$x \sim N(\mu, \sigma^2)$
- Sample $z \sim N(0, 1)$
- Transform $x = \sigma.z + \mu$

### Multi-Dimensional Sampling
`[1]` $x \sim N(0, I)$
Since, `identity matrix` $I$ is independant, we can sample separately.

$$
X = 
\begin{bmatrix}
x_{1} \sim N(0, 1) \\
x_{2} \sim N(0, 1)\\
\vdots \\
x_{N} \sim N(0, 1)
\end{bmatrix}
$$

 `[2]` $x \sim N(\mu, \Sigma)$
 - Sample $z \sim N(0, I)$
 - $x = Lz + \mu$

`Proof`
Suppose $x \sim N(\mu, \Sigma)$ and $y = Ax + b$.
Since `linear transformation` of [[Gaussian Distribution]] is another [[Gaussian Distribution]], we get that
$$
y \sim N(A\mu+b, \ A \ \Sigma A^T)
$$
Then, we can get that 
$$
\begin{align}
&x = Lz + \mu \\[6pt]
&\sim N(L \times 0 + \mu, \ L L^T) \\[6pt]
&\sim N(\mu, \Sigma) \\[6pt]
\end{align}
$$

Note that we use `Cholesky Decomposition` to get $\Sigma = L \ L^T$


---
## Sampling from Categorical Data

`[1]` Find `CDF`
`Example`
$$
PDF =  
\begin{bmatrix}
\underbrace{0.2}_{p(x=1)} & \underbrace{0.5}_{p(x=2)} & \underbrace{0.3}_{p(x=3)}
\end{bmatrix}
$$
Then, we can get that
$$
CDF = \begin{bmatrix}
\underbrace{0.2}_{p(x \ \leq \ 1)} & \underbrace{0.7}_{p(x \ \leq \ 2)} & \underbrace{1}_{p(x \ \leq \ 3)}
\end{bmatrix}
$$

`[2]` Sample $z \sim Uniform(0, 1)$

`[3]` Find the first index $i$ that gives you $CDF(i) > z$

---