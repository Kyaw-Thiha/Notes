# Reconstruction Error

`Centering`
Recall from [[Centering Coordinate System]] that
$$
Y = \begin{bmatrix}
4 & 8 & 12 \\
3 & 6 & 9
\end{bmatrix}
,\quad
\mu_{Y} = \begin{bmatrix}
8 \\ 6
\end{bmatrix}
$$

Then, we get that 
$$
\begin{align}
Z  
&= Y - \mu_{Y} \cdot 1  
\\[6pt]
&= \begin{bmatrix}
-4 & 0 & 4 \\
-3 & 0 & 3
\end{bmatrix}
, \quad \mu_{Z} = \begin{bmatrix}
0 \\ 0
\end{bmatrix}
\end{align}
$$

`Rotation`
Recall from [[Rotating Coordinate System]] that using the orthogonal matrix 
$$
V = \begin{bmatrix}
\frac{4}{5} & -\frac{3}{5}  \\
\frac{3}{5} & \frac{4}{5}
\end{bmatrix}
$$

`Dimension Reduction`
Recall from [[Dimension Reduction]] that 
$$
\hat{X} = \begin{bmatrix}
-5 & 0 & 5
\end{bmatrix}
$$

---
`Reconstruction`

First, we get that
$$
\begin{align}
X &= V^T (Y - \mu_{Y}1) \\[6pt]
VX &= VV^T (Y - \mu_{Y}1) \\[6pt]
VX &= Y - \mu_{Y}1 \\[6pt]
Y - \mu_{Y}1 &= VX
\end{align}
$$

Let $V = \begin{bmatrix}W : W^{\perp}\end{bmatrix}$ and $X = \begin{bmatrix}\hat{X} \\ \dots \\ \hat{X}^{\perp}\end{bmatrix}$.
Then, 
$$
\begin{align}
Y - \mu_{Y}1
&= \begin{bmatrix}
W | W^T
\end{bmatrix}
\begin{bmatrix}
\hat{X} \\ - \\ \hat{X}^{\perp}
\end{bmatrix} \\[6pt]
&= W\hat{X} + W^{\perp} \hat{X}^{\perp} + \mu_{Y}1 \\[6pt]
&= \underbrace{W\hat{X} + \mu_{Y}1}_{\hat{Y}}  
+ W^{\perp} \hat{X}^{\perp}
\end{align}
$$

Using the orthogonal matrix from [[Rotating Coordinate System]], 
$$
\begin{align}
V = \begin{bmatrix}
\frac{4}{5} & -\frac{3}{5}  \\
\frac{3}{5} & \frac{4}{5}
\end{bmatrix}
= \begin{bmatrix}
v_{1} & v_{2}
\end{bmatrix}
= \begin{bmatrix}
W | W^T
\end{bmatrix}
\end{align}
$$

For the `first component`,
$$
\begin{align}
&W = \begin{bmatrix}
\frac{4}{5} \\ \frac{3}{5}
\end{bmatrix}
= \begin{bmatrix}
v_{1}
\end{bmatrix} \\[6pt]

&\hat{X} = \begin{bmatrix}
-5 & 0 & 5
\end{bmatrix} \\[6pt]

&W\hat{X} = \begin{bmatrix}
-4 & 0 & 4 \\
-3 & 0 & 3
\end{bmatrix}
\end{align}
$$

For the `second component`,
$$
\begin{align}
&W^{\perp}
= \begin{bmatrix}
-\frac{3}{5} \\ \frac{4}{5}
\end{bmatrix} 
= \begin{bmatrix}
v_{2}
\end{bmatrix} \\[6pt]

&\hat{X}^{\perp} = \begin{bmatrix}
0 & 0 & 0
\end{bmatrix} \\[6pt]

&W^{\perp} \hat{X}^T = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
\end{align}
$$

Hence we can get that
$$
\begin{align}
\hat{Y}  
&= W\hat{X} + \mu_{Y}1 \\[6pt]
&= \begin{bmatrix}
4 & 8 & 12 \\
3 & 6 & 9
\end{bmatrix} \\[6pt]
&= Y
\end{align}
$$

This means that we have `zero reconstruction error`.

---
