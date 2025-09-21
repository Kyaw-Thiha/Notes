# Linear Regression

Suppose that we are trying to fit **multiple** `line-of-best fits` across different `output features`.
This means that each `line-of-best fit` are multi-dimensional across different `input features`.

Then, we can represent $\tilde{W}$ as a $(N+1) \times K$ matrix.
$$
\begin{align}
\tilde{W} &=  
\begin{bmatrix}
| & \dots & | \\
\tilde{w}_{1} & \dots & \tilde{w}_{K}  \\
| & \dots & | \\
\end{bmatrix}

\\[6pt]

&= \begin{bmatrix}
| & \dots & | \\
w_{1} & \dots & w_{K}  \\
| & \dots & | \\
b_{1} & \dots & b_{K} \\
\end{bmatrix}
\end{align}
$$

Hence, we are essentially solving the multi-dimensional linear equation of
$$
\tilde{y} = \tilde{W}^T.\tilde{x}
$$

## Objective Function
The `objective function` is just to minimize the `squared residual error` over all training samples & output dimensions.

$$
E(\tilde{W}) = \sum^N_{i=1} \sum^K_{j=1} (y_{i, j} - \tilde{w}_{j}^T.\tilde{x}_{i})^2
$$

Note that $(y_{i, j} - \tilde{w}_{j}^T.\tilde{x}_{i})^2$ is [[Multi-Dim Input Linear Regression]] per `output feature`.

So, we stack all $y_{i,j}$, $\tilde{x}_{i}$ and $\tilde{w}^T_{j}$ together.
$$
Y = 
\begin{bmatrix}
y_{i,1} \\
\vdots  \\
y_{i, K}
\end{bmatrix} , 
\
\tilde{X} = 
\begin{bmatrix}
x_{i,1} \\
\vdots  \\
x_{i, K}
\end{bmatrix} , 
\
\tilde{W} = 
\begin{bmatrix}
w_{i,1} \\
\vdots  \\
w_{i, K}
\end{bmatrix}  
$$
And we get
$$
E(\tilde{W}) = || Y - \tilde{X}.\tilde{W} ||^2_{F}
$$
or
$$
E(\tilde{W}) = \sum^K_{j=1}|| y_{j} - \tilde{X}.\tilde{w_{j}} ||^2_{F}
$$
if we want to represent per `output dimension`.

## Solving
Just as we did with [[1D Linear Regression]] and [[Multi-Dim Input Linear Regression]], we can solve 
$$
\tilde{W}^* = \tilde{X}^+.\tilde{y}
$$
where $\tilde{X}^+$ is the `pseudoinverse`

This is equivalent to
$$
\sum^K_{j=1} \tilde{w}^x_{j} = \sum^K_{j=1} \tilde{X}^T.\tilde{y}_{j}
$$
if we want to represent per `output dimension`.

---
## See Also
- [[1D Linear Regression]]
- [[Multi-Dim Input Linear Regression]]
