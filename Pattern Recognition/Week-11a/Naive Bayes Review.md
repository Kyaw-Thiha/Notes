# Naive Bayes
#ml/classic-models/naive-bayes
Consider a classification problem with the [[Bayes Optimal Classifier|Bayes optimal classifier]]:
$$
h_{Bayes}(\mathbf{x})
= \arg\max_{y\in \{ 0,1 \}} p(Y=y \mid X=x)
$$
Enumerating this distribution grows exponentially in the dimensionality of the input.

In [[Naive Bayes]], we assume that each feature is independent, given the label:
$$
p(X=x \mid Y=y) = \prod^{d}_{i=1} p(X_{i} = x_{i} \mid Y = y)
$$
making our classifier to be
$$
\begin{align}
h_{Bayes}(\mathbf{x})
&= \arg \max_{y \in \{ 0,1 \}} p\ (Y=y \mid X=x)  
\\[6pt]

&= \arg \max_{y \in \{ 0,1 \}} \frac{p \ (Y=y)}{\eta}
\ p(X=x  \mid Y=y) \\[6pt]

&= \arg \max_{y \in \{ 0,1 \}} p(Y=y) \
\prod^{d}_{i=1} p(X_{i} = x_{i} \mid Y=y)
\end{align}
$$

[[Naive Bayes]] enforces the complexity of our distribution to be linear in the number of features.

Note that it is making a simple assumption about how to factorize a probability distribution.

---
## See Also
- [[Naive Bayes]]
- [[Gaussian CCM]]
- [[Bayes Optimal Classifier]]