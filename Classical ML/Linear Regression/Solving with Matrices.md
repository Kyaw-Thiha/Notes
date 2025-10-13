# Solving with Matrices
#ml/classic-models/linear-regression/solving/matrices #math
Using augmented matrices $\tilde{x} = \begin{pmatrix}1 \\ \vec{x}\end{pmatrix}$ and $\tilde{w} = \begin{pmatrix}b \\ \vec{w}\end{pmatrix}$,
$$
\tilde{X} = 
\begin{bmatrix}
1 & x_{1}  \\
1 & x_{2} \\
\vdots \\
1 & x_{N}
\end{bmatrix}
= 
\begin{bmatrix}
\tilde{x}^T_{1} \\
\tilde{x}^T_{2}  \\
\vdots \\
\tilde{x}^T_{N}  \\
\end{bmatrix}
\ , \
\tilde{w} = 
\begin{bmatrix}
b \\ 
w_{1} \\
\vdots \\
w_{n} \\
\end{bmatrix}
$$
such that $\tilde{X}$ is $N \times (D+1)$ and $\tilde{w}$ is $(D+1) \times 1$.

we plan to minimize the predicted values $\hat{y}$ to ground truth $y$
$$
y = 
\begin{bmatrix}
y_{1} \\
y_{2} \\
\vdots \\
y_{N}
\end{bmatrix}
\ , \
\hat{y} = 
\begin{bmatrix}
\hat{y}_{1} \\
\hat{y}_{2} \\
\vdots \\
\hat{y}_{N}
\end{bmatrix}
\ , \
\vec{v} = 
\begin{bmatrix}
v_{1} \\
v_{2} \\
\vdots \\
v_{N}
\end{bmatrix}
=
\begin{bmatrix}
y_{1} - \hat{y}_{1} \\
y_{2} - \hat{y}_{2} \\
\vdots \\
y_{N} - \hat{y}_{N} \\
\end{bmatrix}
$$

This means that 
- The `line-of-best fit` equation is
$$
\hat{f}(x) = w.x + b
$$
- Predicted value $\hat{y_{i}}$ can be solved by 
$$
\hat{y}_{i} = \tilde{x}_{i}.\tilde{w}
$$

So, the [[Loss Function]] can be derived as
$$
\begin{align}
E(\tilde{w}) 
&= \sum^N_{i=1} (w.x_{i} + b - y_{i}) ^ 2 \\
&= \sum^N_{i=1} (\hat{y}_{i} - y_{i}) ^ 2 \\
&= \sum^N_{i=1} (\vec{v}_{i}) ^ 2 \\
&= \vec{V}^T.\vec{V} \\
&= ||\vec{V}||^2 \\
\end{align}
$$

Continuing on that
$$
\begin{align}
E(\tilde{w}) 
&= \lVert \vec{y} - \hat{y} \rVert^2 \\
&= (\vec{y} - \hat{y})^T \cdot (\vec{y} - \hat{y}) \\
&= (\vec{y} - \tilde{X}\tilde{w})^T \cdot (\vec{y} - \tilde{X}\tilde{w}) \\ 
&= (\vec{y}^T - \tilde{w}^T \tilde{X}^T) \cdot (\vec{y} - \tilde{X}\tilde{w}) \quad \quad \quad \text{since }(A.B)^T = B^T.A^T \\ 
&= \vec{y}^T \cdot \vec{y} - \tilde{w}^T \tilde{X}^T \cdot \vec{y} - \vec{y}^T \cdot \tilde{X}\tilde{w} + \tilde{w}^T \tilde{X}^T \tilde{X}\tilde{w} \\ 
&= \vec{y}^T \cdot \vec{y} - 2 \cdot \vec{y}^T \tilde{X}\tilde{w} + \tilde{w}^T \tilde{X}^T \tilde{X}\tilde{w}
\end{align}
$$

To optimize that and find the minimum, we need to find the critical point using derivatives.
Note that 
- $\frac{\partial}{\partial w}\vec{y}^T.\vec{y} = 0$ as it is constant
- Consider $S = \tilde{X}^T.\tilde{X}$ as it is symmetric.
- So, we can use [[Vector Calculus (For ML)]]
$$
\begin{align}
\frac{\partial E}{\partial \tilde{w}}
&= 0 \\
-2(\tilde{X}^T.\vec{y}) + 2\tilde{X}^T.\tilde{X}.\tilde{w}
&= 0 \\
\tilde{X}^T.\tilde{X}.\tilde{w}
&= \tilde{X}^T.\vec{y} \\
\tilde{w}
&= (\tilde{X}^T.\tilde{X})^{-1}.\tilde{X}^T.\vec{y} \\
\end{align}
$$

where $(\tilde{X}^T.\tilde{X})^{-1}.\tilde{X}^T$ is the `pseudo-inverse`.
