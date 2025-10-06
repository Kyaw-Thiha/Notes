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
## Optimizing Objective Function
$$
E(W, b) = \sum^N_{i=1} (\hat{y}_{i} - y_{i})^2
$$
$$
\frac{\partial E}{\partial b}  
= \frac{\partial}{\partial b} \sum^N_{i=1} (wx_{i} + b - y_{i})^2 \\ 
$$
$$
\begin{align}
\frac{\partial E}{\partial b}  &= 0 \\
2. \sum^N_{i=1} (wx_{i} + b - y_{i}) &= 0 \\
\sum^N_{i=1} (wx_{i} - y_{i}) + \sum^N_{i=1}b &= 0 \\
w.\sum^N_{i=1} x_{i} - \sum^N_{i=1} y_{i} + Nb &= 0
\end{align}
$$

