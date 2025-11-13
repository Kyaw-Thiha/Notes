# Class-Conditional Model (CCM)

`Class-Conditional Model` model the distribution over features themselves for each class.
$$
p(x, y) = p(y) \ p(x|y)
$$

- The `prior` $p(y)$ is relatively easy to learn.
- The `class-conditional probability` require usage of [[Bayes Rule in ML]] to compute the posterior $p(y|x^*)$
$$
p(y|x^*) = \frac{p(x^*|y) \ p(y)}{p(x)}
$$
where $x^*$ is the unseen data we want to classify
