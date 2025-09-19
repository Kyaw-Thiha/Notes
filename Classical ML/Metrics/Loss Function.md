## Loss Function
Loss function means the error of the predicted value.
Distance between the predicted value and the true value.

$$
L(y_{i}, \hat{y}_{i})
$$
Squared Error: $e_{i}^2 = (y_{i} - \hat{y}_{i})^2$
Absolute Error: $|e_{i} = (y_{i} - \hat{y}_{i})|$

## Cost Function
Cost function is mean of loss function over all data points & their corresponding predictions.
$$
J(\theta) = \frac{1}{N} \sum_{i=1}^N L(y_{i}, \hat{y}_{i})
$$
$$
E(\theta) = \sum_{i=1}^N L(y_{i}, \hat{y}_{i})
$$
