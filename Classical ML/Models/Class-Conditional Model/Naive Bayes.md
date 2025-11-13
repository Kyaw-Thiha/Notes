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
