# Bayes Classifier
#ml
`Bayes Classifier` is the theoretical best possible classifier you can have.

It classifies by choosing the class with highest `posterior probability`:
$$
\hat{y} = \arg \max_{c} p(y = c \ | \ x)
$$

---
## Binary Classification

For `Binary Classification` where $y \in \{ c_{0}, c_{1} \}$, let the true posterior be 
$$
\begin{align}
r(x^*)  
&= P(y= c_{0} | x = x^*) \\[6pt]
&= \frac{P(x = x^* | y = c_{0}) \ P(y = c_{0})} 
{P(x = x^*)} \\[6pt]
&= \frac{P(x = x^* | y = c_{0}) \ P(y = c_{0})} 
{P(x = x^* | y=c_{0}) \ P(y = c_{0}) + P(x = x^* | y=c_{1}) \ P(y = c_{1})} \\[6pt]
\end{align}
$$

To choose the class with highest `posterior probability`,
$$
h(x^*)
= \begin{cases}
c_{0}, \text{ if } r(x^*) > \frac{1}{2}  \\
c_{1}, \text{ otherwise}
\end{cases}
$$

The [[Bayes Rule]] is optimal.
This means the true error $1 - r(x^*)$ of `Bayes Classifier` is the smallest.

---
## Types
### Generative Bayes Classifier
These classifiers explicitly model the `class-conditional densities` $p(x|y)$ and `class priors` $p(y)$, before applying [[Bayes Rule]]
$$
p(y \ | \ x) \propto p(x \ | \ y).p(y)
$$

`Eg`:
- `Gaussian Naive Bayes`
- `Full Gaussian Bayes`

### Discriminative Bayes Classifier
These classifiers directly model
$$
p(y \ | \ x)
$$
This is what modern `neural networks` do.

---
## See Also
- [[Class-Conditional Model]]
- [[Binary Classification CCM]]
