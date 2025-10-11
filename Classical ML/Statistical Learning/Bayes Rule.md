# Bayes Rule
#ml/bayes-rule #math

This is considered `Generative Model`
$P(X, Y) = P(X|Y).P(Y)$

This is considered `Discriminative Model`
$P(X, Y) = P(Y|X).P(X)$

### Bayes Rules Variation
Let $K$ be discrete and $\Theta$ be continuous variables.

Then,
$$
\begin{align}
&P(K = k, \theta \leq \Theta \leq \theta + \delta) \\[6pt]
= &P(K = k).P(\theta \leq \Theta \leq \theta + \delta | K = k) \\[6pt]
= &P(\theta \leq \Theta \leq \theta + \delta).P(K=k | \theta \leq \Theta \leq \theta + \delta) \\[6pt]
\end{align}
$$

Alternatively,
$$
\begin{align}
&P(K = k, \theta \leq \Theta \leq \theta + \delta) \\[6pt]
= \quad &p_{K}(k).p_{\Theta | K} (\theta | k).\delta \\[6pt]
= \quad &p_{\Theta}(\theta).p_{K | \Theta} (k | \theta).\delta \\[6pt]
\end{align}
$$

Equivalently,
$$
\begin{aligned}
p_{\Theta \mid K}(\theta \mid k) = \frac{p_{\Theta}(\theta)\, p_{K \mid \Theta}(k \mid \theta)}{p_K(k)}
\quad\quad
p_{K \mid \Theta}(k \mid \theta) = \frac{p_K(k)\, p_{\Theta \mid K}(\theta \mid k)}{p_{\Theta}(\theta)}
\end{aligned}
$$

In ML Literature, when the meaning of the random variables are obvious from context, 

$$
\begin{aligned}
p(\theta \mid k) = \frac{p(\theta)\, p(k \mid \theta)}{p(k)}
\quad\quad
p(k \mid \theta) = \frac{p(k)\, p(\theta \mid k)}{p(\theta)}
\end{aligned}
$$

