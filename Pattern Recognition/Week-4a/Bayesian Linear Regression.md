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