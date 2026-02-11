# Bayesian Linear Regression
`Bayesian Linear Regression` can be thought of as [[Linear Regression]] that computes the expectation over all possible `weight vectors`.

![Bayesian Linear Regression|400](https://gregorygundersen.com/image/linbayes/bishop_3.7.png)


![Bayesian Linear Regression|400](https://krasserm.github.io/img/2019-02-23/output_8_0.png)

---

The [[Maximum Likelihood Estimation (MLE)|MLE]] and [[Maximum A Posteriori (MAP)|MaP]] for [[Linear Regression|linear regression]] can be thought of as estimating specific points in the distribution.

In `Bayesian linear regression`, we instead compute the expectation over all possible weight vectors.
$$
p(y \mid x, S)
= \int p(y \mid x,w) \ p(w \mid S) \ dw
$$

Bayesian updating of [[Gaussian Distribution]] can be used to analytically derive the normal distribution of the form
$$
p(y \mid x, S)
= \mathcal{N}(y \mid m(x), s^2(x))
$$
where
- $m(x) = \beta \ x^T S \sum^m_{i=1} x_{i}\ y_{i}$ 
- $s^2(x) = \beta^{-1} + x^TSx$
- $S^{-1} = \alpha I + \beta \sum^m_{i=1} \times^T$

in which $\alpha$ and $\beta$ defined by the `MaP estimator`

---
## MLE Solution
In order to get the `MLE solution`, our [[Log Likelihood|Negative Log Likelihood]] function becomes
$$
-\log p(S \mid \mathbf{w}, \beta)
= \frac{\beta}{2} \sum^m_{i=1} 
(y_{i} - \langle \mathbf{w}, \mathbf{x}_{i} \rangle)^2
- \frac{m}{2} \log \beta
+ \frac{m}{2} \log 2\pi
$$

which can be solved by standard Maximum Likelihood technique.

Note that when we optimize $w.r.t$ $\mathbf{w}$,
1. Only the first term matters
2. Since $\beta > 0$, it will not effect which $w$ is optimal
3. Solving the negative log likelihood is equivalent to solving the [[Empirical Risk Minimization (ERM)]]

---
## MAP Solution
To get a `MAP Solution`, place a prior distribution over our weights $\mathbf{w} \in \mathbb{R}^d$, conditioned by a hyperparameter $\alpha$.

$$
\begin{align}
&p(\mathbf{w} \mid \alpha) \\[6pt]
&= \mathcal{N}(\mathbf{w} \mid 0,  
\alpha^{-1} I) \\[6pt]
&= \left( \frac{\alpha}{2\pi} \right)^{(d+1)/2}
\exp \left\{  -\frac{\alpha}{2} \mathbf{w}^T \mathbf{w}  \right\}
\end{align}
$$

Hence, the posterior can be rewritten as
$$
p(\mathbf{w} \mid x, y, \alpha, \beta)
\quad \propto \quad
p(\mathbf{y} \mid x, w, \beta) 
\ p(\mathbf{w} \mid \alpha)
$$

This makes the objective function to be
$$
\begin{align}
&-\log p(S \mid \mathbf{w}, \beta)
\ p(\mathbf{w} \mid \alpha) \\[6pt]
&= \frac{\beta}{2} \sum^m_{i=1}
(y_{i} - \langle w, x_{i} \rangle)^2
- \frac{m}{2} \log \beta
+ \frac{m}{2} \log 2\pi  \\
&+ \frac{\alpha}{2} w^Tw
- \frac{d+1}{2} \ \log (\frac{\alpha}{2\pi})
\end{align}
$$

which can be solved by standard MAP technique.

---