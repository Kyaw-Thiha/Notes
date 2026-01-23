# 1D Linear Regression
#ml/models/classic/linear-regression/1-dim-input

This is the most trivial case of linear regression.
It can be best understood as trying to get a line-of-best fit.

![Linear Regression](https://media.geeksforgeeks.org/wp-content/uploads/20231129130431/11111111.png)

## How to Mathematically Compute

- Suppose we have a set of $(x_{i}, y_{i})$ data points.
- Then, let the error $e_{i} = y_{i} - (w.x_{i} + b)$ be the vertical distance between ground-truth & predicted value.
- The sum of the squared errors is used to fit the data
$$
E(w, b) = \sum^N_{i=1}e_{i}^2 = \sum^N_{i=1}(y_{i} - (w.x_{i} + b))^2
$$
- To minimize the squared error, we find the critical point (minimum)
$$
\begin{align}
\frac{\partial E}{\partial b} 
&=  0 \\[8pt]
-2 \sum_{i}^N \bigl(y_i - (w x_i + b)\bigr) &= 0  \\
\sum_{i=1}^N \big(y_i - w x_i - b\big) &= 0 \\
\sum_{i=1}^N y_i - w \sum_{i=1}^N x_i - \sum_{i=1}^N b &= 0 \\
\sum_{i=1}^N y_i - w \sum_{i=1}^N x_i - N b &= 0 \\
\sum_{i=1}^N y_i - w \sum_{i=1}^N x_i &= N.b \\
\frac{1}{N}\sum_{i=1}^N y_i - w \cdot \frac{1}{N}\sum_{i=1}^N x_i &= b^*
\end{align}
$$

- Define the averages, and get a better formula 
$$
\hat{x} = \frac{1}{N}\sum_{i=1}^N x_i, 
\quad 
\hat{y} = \frac{1}{N}\sum_{i=1}^N y_i
$$
$$
b^* = \hat{y} - w \hat{x}
$$
	It can be thought of that the `line-of-best-fit` pass through all $(\hat{x}, \hat{y})$.

- Now, we substitute the $(\hat{x}_{i}, \hat{y}_{i})$, but use them for 'centering' the data points $(x_{i}, y_{i})$ to get the `energy function`.
$$
\begin{align}
E(w, b) &= \sum^N_{i=0}(y_{i} - (w.x_{i} + b^*))^2 \\
E(w, b) &= \sum^N_{i=0}[y_{i} - (w.x_{i} + (\hat{y} - w.\hat{x}))]^2 \\
E(w, b) &= \sum^N_{i=0}((y_{i} - \hat{y}) - w.(x_{i} - \hat{x}))^2 \\
\end{align}
$$

- Differentiate and take the zero derivative.

$$
\begin{aligned}
\frac{\partial E}{\partial w} &= 0  \\

\sum_{i=1}^N 2\Big((y_i-\hat{y}) - w(x_i-\hat{x})\Big)\cdot \frac{\partial}{\partial w}\Big((y_i-\hat{y}) - w(x_i-\hat{x})\Big) &= 0
 \\

\sum_{i=1}^N \Big((y_i-\hat{y}) - w(x_i-\hat{x})\Big)\,.(x_i-\hat{x}) &= 0 \\

\sum_{i=1}^N (y_i-\hat{y})(x_i-\hat{x})
- w \sum_{i=1}^N (x_i-\hat{x})^2
&= 0
  \\
w^* \;=\; \frac{\sum_{i=1}^N (y_i-\hat{y})(x_i-\hat{x})}
{\sum_{i=1}^N (x_i-\hat{x})^2}
\end{aligned} 
$$

- Once $w^*$ is known, recover the intercept $b^*$:
$$
b^* \;=\; \hat{y} - w^* \hat{x}.
$$
