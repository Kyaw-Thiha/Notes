# Multi-Dim Input Linear Regression
#ml/models/classic/linear-regression/multi-dim-input
Suppose that we are trying to fit a `line-of-best fit` over multi-dimension inputs.
Then, we will have $n$ number of gradients + 1 bias to fit.

$$
\begin{align}
f(x) &= w^T.x + b \\
f(x) &= \sum^D_{j=1}w_{j} .x_{j} + b
\end{align}
$$

For convenience, we concatenate `b` onto `w` and augment `x`.
$$
\tilde{w} =
\begin{bmatrix}
w_1 \\
\vdots \\
w_D \\
b
\end{bmatrix}
, \
\tilde{x} =
\begin{bmatrix}
x_1 \\
\vdots \\
x_D \\
1
\end{bmatrix}
$$

This way, the `line-of-best fit` can be written as 
$$
f(x) = \tilde{w}^T.\tilde{x}
$$

Hence, the mean squared error can be represented as 
$$
E(\tilde{w}) = \sum^N_{i=1}(y_{i} - \tilde{w}^T.\tilde{x}_{i})^2
$$
Representing the equation using matrices,
$$
E(\tilde{w}) = ||y - \tilde{X}.\tilde{w}||^2
$$
where
$$
y =
\begin{bmatrix}
y_1 \\
\vdots \\
y_{N} 
\end{bmatrix}
, \
\tilde{X} =
\begin{bmatrix}
X_{1}^T & 1 \\
\vdots \\
X_{N}^T & 1
\end{bmatrix}
$$
The equation $E(\tilde{w}) = ||y - \tilde{X}.\tilde{w}||^2$ is called a `linear least squares problem`.
We can rewrite it as
$$
\begin{align}
E(w) &= (y - \tilde{X} \,\tilde{w})^{T} (y - \tilde{X}\,\tilde{w}) \\[6pt]
&= \tilde{w}^{T} \tilde{X}^{T} \tilde{X}\,\tilde{w}
- 2 y^{T} \tilde{X}\,\tilde{w}
+ y^{T}y
\end{align}
$$
Using [[Normal Equation Derivation]], we can get
$$
\tilde{w}^* = (\tilde{X}^T.\tilde{X})^{-1}.\tilde{X}^T.y
$$
This is the equation that can be solved since we already know the values of `X` and `y`.

Note that the matrix $\tilde{X}^+ =  (\tilde{X}^T.\tilde{X})^{-1}.\tilde{X}^T$ is called a `pseudoinverse`.
So, we can rewrite it as $\tilde{w}^* = \tilde{X}^+.y$ and solve it.