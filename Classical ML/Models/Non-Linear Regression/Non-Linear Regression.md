# Non-Linear Regression
#ml/models/classic/non-linear-regression

For data with non-linear relation, we can use `Non-Linear Regression`.

![Linear vs Non-Linear](https://media.geeksforgeeks.org/wp-content/uploads/20240614192441/Linear-Regression-vs-Polynomial-Regression.webp)

Applying `Non-Linear Regression` is also considered as `feature engineering`.
One good way to see this is that we are converting the features before applying normal [[Linear Regression]].

To carry out `Non-Linear Regression`,  
- Given input $X^{N \times K}$ where $N$ is the number of data and $K$ is the number of features, we convert it into $\hat{X}^{N \times K'}$ where $K'$ is the length of new features.
- We get this set of new features by applying either `Polynomial Function` or `Radial Basis Functions` on the data
- From this, we will be using the same `Pseudo-Inverse` formula from [[Solving with Matrices]]

## Types of Non-Linear Regression
There are 2 main types of `Non-Linear Regression`.
- [[Polynomial Regression]]: Better suited for data with long-range global relation (Eg: time-series financial data)
- [[RBF Regression]]: Better suited for data with short-range local relation (Eg: images)

## See Also
- [[Linear Regression]]
- [[Polynomial Regression]]
- [[RBF Regression]]
