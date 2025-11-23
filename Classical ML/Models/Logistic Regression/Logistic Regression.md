# Logistic Regression
#ml/models/classic/logistic-regression  

`Logistic regression` is a binary classifier that uses the [[Sigmoid Function]].

![Logistic Regression](https://miro.medium.com/v2/resize:fit:1400/1*2IAHsCEZaZtZ9ybziYPDEQ.jpeg)

---
## Deriving the Sigmoid Function

Recall from [[Class-Conditional Model]] that we have
$$
P(c_{k}|x)
= \frac{P(c_{k}, x)}{P(x)}
= \frac{P(x|c_{k}) \ P(c_{k})}{\sum^K_{j=1} P(x|c_{j}) \ P(c_{j})}
$$
and in the case of [[Binary Classification CCM|Binary Classification]], we have
$$
P(c_{1}|x) = \frac{P(x|c_{1}) \ P(c_{1})}{P(x|c_{1}) \ P(c_{1}) + P(x|c_{2}) \ P(c_{2})}
$$

Divide both the numerator & denominator with $P(x|c_{1})P(c_{1})$
$$
P(c_{1}|x)
= \frac{1}{1 + \frac{P(x|c_{2}) \ P(c_{2})}{P(x|c_{1}) \ P(c_{1})}}
$$
Using the [[Binary Classification CCM#Decision Function|Decision Function]] from [[Binary Classification CCM]],
$$
\begin{align}
a(x) 
&= \log\left( \frac{P(x|c_{1}) \ P(c_{1})}{P(x|c_{2}) \ P(c_{2})} \right) \\[6pt]
e^{a(x)} &=  \frac{P(x|c_{1}) \ P(c_{1})}{P(x|c_{2}) \ P(c_{2})}  \\[6pt]

e^{a(x)} &=  \frac{P(c_{1}|x)}{P(c_{2}|x)}  
&\text{by Bayes Rule} \\[6pt]

e^{a(x)} &=  \frac{P(c_{1}|x)}{1 - P(c_{1}|x)}  
&\text{by Law of Total Probability} \\[6pt]
\end{align}
$$

Continuing on,
$$
\begin{align}
e^{a(x)} &=  \frac{P(c_{1}|x)}{1 - P(c_{1}|x)}  \\[6pt]
P(c_{1}|x) &= e^{a(x)} (1 - P(c_{1}|x)) \\[6pt]
P(c_{1}|x) &= e^{a(x)} - e^{a(x)}.P(c_{1}|x) \\[6pt]
P(c_{1}|x) + e^{a(x)}.P(c_{1}|x) &= e^{a(x)} \\[6pt]
P(c_{1}|x) ( 1 + e^{a(x)}) &= e^{a(x)} \\[6pt]
P(c_{1}|x) &= \frac{e^{a(x)}}{1 + e^{a(x)}} \\[6pt]
P(c_{1}|x) &= \frac{1}{1 + e^{-a(x)}} \\[6pt]
\end{align}
$$

---
## Linear Decision Boundary

`Assumption`
Assume that the decision boundary is linear.
$$
a(x) = w^T\ x
$$

Then, we obtain the posterior as
$$
P(c_{1}|x) = \frac{1}{1 + e^{-w^Tx}}
$$

`Remarks`
- For `Exponential Distribution Family`, the `decision boundary` is always linear.
- Note that we only assuming `posterior` to be [[Gaussian Distribution]].
  The `likelihood` and `prior` can be either `Gaussian` or `Non-Gaussian`.
- Recall that in [[Gaussian CCM]], `decision boundary` is linear when $\Sigma_{1} = \Sigma_{2}$.

---
## No. of Parameters
Let dimension of input data be $D$.

[[Gaussian CCM]]
$D$ parameters for `mean` and $\frac{D(D+1)}{2}$ parameters for the covariance matrix.
Total: $D + \frac{D(D+1)}{2}$

[[Naive Bayes]]
$D$ parameters for `mean` and $D$ parameters for `diagonal covariance matrix`.
Total: $2D$

[[Logistic Regression]]
Only needs to learn $D+1$ parameters.

---
## Learning Logistic Regression
We can learn the optimal parameters of `Logistic Regression` by using numerical methods like [[Gradient Descent]].

[[Learning Logistic Regression|Read More about learning Logistic Regression]]

---
## Regularized Logistic Regression
We can regularize `Logistic Regression` against overfitting by penalizing large values of $w$.
$$
\tilde{E}(w) = E(w) + \lambda w^Tw
$$

This means when differentiating, we get
$$
\begin{align}
\nabla \tilde{E}(w)
&= \nabla E(w) + \nabla (\lambda w^Tw) \\[6pt]
&= -\sum^N_{i=1} (y_{i} - g(w^T x_{i})) \ x_{i} + 2\lambda w
\end{align}
$$

---
## Sigmoid vs Soft-Max
For `Binary Classification` tasks, we use [[Sigmoid Function]].
For `Multi-Class Classification` tasks, we use [[Softmax Function]].

---
## See Also
- [[Class-Conditional Model]]
- [[Sigmoid Function]]
- [[Learning Logistic Regression]]