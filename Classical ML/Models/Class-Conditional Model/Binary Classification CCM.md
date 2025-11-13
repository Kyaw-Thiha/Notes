# Binary Classification CCM

In order to carry out binary classification with [[Class-Conditional Model]], we first model the probability distribution over features $x = [x_{1}, x_{2}]^T$ for both classes.
$$
P(x | c_{1}) \quad P(x | c_{2})
$$

Then given $x^* = [x_{1}^*, x_{2}^*]^T$, evaluate the `posterior probability`.
$$
\begin{align}
&p(y = c_{1}|x^*) > p(y = c_{2}|x^*) \\[6pt]
&\frac{p(y = c_{1}|x^*)}{p(y = c_{2} | x^*)} > 1
\end{align}
$$

---
## Learning
Given data $\{ (x_{i}, y_{i}) \}^N_{i=1}$, $y_{i} \in \{ c_{1}, c_{2} \}$,
- Use fraction of data with label $c_{1}$ as $p(c_{1})$
- Partition dataset $x_{i}$ into $c_{1}$, $c_{2}$
  Use all $x_{i} \text{ s.t. } y_{i} = c_{1}$  to learn $p(x \ | \ y = c_{1})$
  Use all $x_{i} \text{ s.t. } y_{i} = c_{2}$  to learn $p(x \ | \ y = c_{2})$
  
We can then assume [[Gaussian Distribution]] of the `likelihood`
$$
P(x | y = c_{i}) = G(x; \mu_{i}, \Sigma_{i})
$$
where
- mean vector $\mu_{i}$
- covariance matrix $\Sigma_{i}$
are `parameters` to be learned

---
## Decision Function
First, apply the [[Bayes Rule]],
$$
\begin{align}
\frac{p(y = c_{1} | x)}{p(y = c_{2} | x)}
&= \frac{p(c_{1}) \ p(x|c_{1})}{p(x)} \times \frac{p(x)}{p(c_{2}) \ p(x|c_{2})} \\[6pt]
&= \frac{p(c_{1}) \ p(x|c_{1})}{p(c_{2}) \ p(x|c_{2})} \\[6pt]
&= \frac{p(x|c_{1})}{p(x|c_{2})} \times \frac{p(c_{1})}{p(c_{2})} \\[6pt]
&= \text{ratio of likelihood } \times \text{ ratio of priors}
\end{align}
$$

The `decision function` is the log of the product of the two ratios
$$
\begin{align}
&\frac{p(x|c_{1})}{p(x|c_{2})} \times \frac{p(c_{1})}{p(c_{2})} > 1 \\[12pt]
&a(x) = \log\left( \frac{p(x|c_{1})}{p(x|c_{2})} \times \frac{p(c_{1})}{p(c_{2})} \right) > 0 \\[12pt]
\end{align}
$$

The `classifier`:
$$
sgn(a(x)) =  
\begin{cases}
-1, \quad  a(x) \leq 0 \\[6pt]
1, \quad a(x) > 0
\end{cases}
$$

The `decision boundary` is $a(x) = 0$

---
