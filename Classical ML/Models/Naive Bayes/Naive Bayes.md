# Naive Bayes
`Naive Bayes` is [[Gaussian CCM]] that uses assumption that covariance matrix is diagonal to reduce the number of parameters.

![Naive Bayes](https://miro.medium.com/v2/1*ZW1icngckaSkivS0hXduIQ.jpeg)

---
## Assumptions
- Ignore correlation between features
- Features are conditionally independant given class label
- `Covariance Matrix` is diagonal

These assumptions reduce the total number of parameters need to learn to $O(D)$

Hence,
$$
p(x|c) = p(x_{1}, x_{2}, \dots, x_{D} | C)
= \Pi^D_{j=1} p(x_{j} | c)
$$

---
### Non-Naive Bayes Parameters
Let features $D = 3$.
$$
\begin{align}
P(F_{1:D} | c_{1})
&= P(F_{1}, F_{2}, F_{3} \ | \ c_{1}) \\[6pt]
&= P(F_{1} \ | \ F_{2} F_{3} c_{1}) \ P(F_{2}, F_{3} | c_{1}) \\[6pt]
&= P(F_{1} \ | \ F_{2} F_{3} c_{1}) \ P(F_{2} | F_{3}c_{1}) \ P(F_{3}|c_{1}) \\[6pt]
\end{align}
$$
- $P(F_{3}|c_{1})$
  This is a `Bernoulli Distribution`, which requires `1 parameter`: $P(F_{3} = 0|c_{1}) = \theta_{1}$ and $P(F_{3} = 1|c_{1}) = 1 - \theta_{1}$
- $P(F_{2} | F_{3}c_{1})$
  Since there are 2 possible conditions of $(F_{3},c_{1})$, we need `2 parameters`
  - $P(F_{2} = 0 | F_{3} = 0, c_{1}) = \theta_{2}$ and $P(F_{2} = 1 | F_{3} = 0, c_{1}) = 1 - \theta_{2}$
  - $P(F_{2} = 0 | F_{3} = 1, c_{1}) = \theta_{3}$ and $P(F_{2} = 1 | F_{3} = 1, c_{1}) = 1 - \theta_{3}$
- $P(F_{1} F_{2} F_{3} | c_{1})$
  Since there are 4 possible combinations of $(F_{2}, F_{3}, c_{1})$, we need `4 parameters`
  - $P(F_{1} = 0 | F_{2} = 0, F_{3} = 0, c_{1})= \theta_{4}$
  - $P(F_{1} = 0 | F_{2} = 0, F_{3} = 1, c_{1})= \theta_{5}$
  - $P(F_{1} = 0 | F_{2} = 1, F_{3} = 0, c_{1})= \theta_{6}$
  - $P(F_{1} = 0 | F_{2} = 1, F_{3} = 1, c_{1})= \theta_{7}$

This mean we need a total of $2^D-1 = 7$ parameters.
Note that $2^D - 1$ implies that no. of parameters grow `exponentially` with feature dimension.

### Naive-Bayes Parameters
Since features are `conditionally independent`, 
$$
P(F_{1} F_{2} F_{3} | c_{1}) = \Pi^3_{i=1} P(F_{i} | c_{1})
$$
Hence, we only need to estimate `3 parameters`.
This means that no. of parameters grows `linearly` with feature dimension now.

---
## Naive-Bayes Classification
Let classes $c \in \{ 1, 2, \dots, C \}$
Then,
$$
\begin{align}
&P(c = j | F_{1:D}) \\[6pt] \\

&= \frac{P(F_{1:D} | c=j) \ .P(c=j)}{P(F_{1:D})}
& \text{ by Bayes Rule} \\[6pt] \\

&= \frac{\Pi^D_{i=1} \ P(F_{i} | c=j) \ .P(c=j)}{P(F_{1:D})}
& \text{ by Conditional Independance} \\[6pt] \\

&= \frac{\Pi^D_{i=1} \ P(F_{i} | c=j) \ .P(c=j)}{\sum^C_{k=1} P(F_{1:D}, c=k)}
& \text{ by Marginalizing Rule} \\[6pt] \\
\end{align}
$$

Continuing on, note that
- $$
\begin{align}
&\Pi^D_{i=1} \ P(F_{i} | c=j)  \\[6pt]
&= \Pi^D_{i=1} P(F_{i} = 1 | c = j) \ . \ \Pi^D_{i=1} P(F_{i} = 0 | c = j)
\end{align}
$$
- $$
\begin{align}
  &P(F_{1:D} | c=k)  \\[6pt]
&= \Pi^D_{i=1} P(F_{i}, c = k)   \\[6pt]
&= \Pi^D_{i=1} P(F_{i} = 1 | c = k) \ . \ \Pi^D_{i=1} P(F_{i} = 0 | c = k)
\end{align}
  $$
Hence, it can be written as
$$
\frac{\Pi^D_{i=1} P(F_{i} = 1 | c = j) \ . \ \Pi^D_{i=1} P(F_{i} = 0 | c = j) \ P(c=j)}
{\sum^C_{k=1} [\Pi^D_{i=1} P(F_{i} = 1 | c = k) \ . \ \Pi^D_{i=1} P(F_{i} = 0 | c = k)] \ P(c=k)}
$$

---
## Remarks
- For [[Estimation#`MAP (Maximum A Posteriori)`|MAP]], we can ignore the denominator.
- We usually compute using [[Log Likelihood]], since $P(.)$ can be small, which can cause [[Floating Points|Underflow]]

---
## Naive-Bayes Learning
Given $N$ training vector $F_{i}$, each with class label $c_{i}$, 
we can learn its parameters by [[MLE for Gaussian Distribution]].

This is equivalent to estimating the `Multinomial Distribution`, hence reducing it to counting the no. of features.
- `Class Prior`: $P(c=j) = \frac{N_{j}}{N} = \frac{\text{No. of Training Samples with class }j}{\text{Total no. of training samples}}$
- `Conditional Feature Probability`: $P(F_{i} = 1|c=j) = \frac{N_{i,j}}{N_{j}} = \frac{\text{No. of Training Samples with class label j where } F_{1}=1}{\text{No. of training samples with class label j}}$

## Bayes Estimator
[[Bayesian Estimation]] is useful for small dataset with large no. of features.
This is because some features will never be seen for some classes.

Let $\alpha, \beta$ be small constants reflecting degree of uncertainty.
Then, 
- `Class Prior`: $P(c=j) = \frac{N_{j} + \beta}{N + \beta C}$
- `Conditional Feature Probability`: $P(F_{i} = 1| c=j) = \frac{N_{i,j} \ + \ \alpha}{N_{j} \ + \ 2\alpha}$

---
## See Also
- [[Gaussian CCM]]
- [[Class-Conditional Model]]
- [[Logistic Regression]]
