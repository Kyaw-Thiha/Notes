# Expected Test Error Given Dataset D

Having multiple values of `y` for a given `x`, allow us to denote the mean $\bar{y}$ as [[Expected Target Value]].

![[Expected Test Error.png]]

`Expected Test Error` is the value computed by measuring how different our predicted $\hat{y}_{D}$ trained on dataset $D$ is compared to our [[Expected Target Value]].

$$
E_{X, Y} \left[ L(\hat{y}_{D}, \ y) \right] 
= \int_{x} \ \int_{y} \ L(\hat{y}_{D}(x), \ y).p(x, y) \ dy dx
$$
where
- $\hat{y}_{D} = w_{D}^T x$  is the predicted value for dataset $D$
- $y$ is the `Expected Target Value`

## Squared Expected Test Error
If we use squared loss, our `Expected Test Error` becomes
$$
E_{X, Y} \left[ L(\hat{y}_{D}, \ y) \right] 
= \int_{x} \ \int_{y} \ (\hat{y}_{D}(x) - y)^2.p(x, y) \ dy dx
$$
## Expected Predictor
We can define `Expected Predictor` of a model as
$$
\bar{\hat{y}} = E_{D} [\hat{y}_{D}] 
= \int_{D} \hat{y}_{D}(x).p(D) \ dD
$$

In other words, it is the sum of different predicted values times the probability of those datasets occurring.
