# Perturbed Reconstruction Error

Coming from [[Reconstruction Error]], lets take at look at a perturbed dataset

![[PCA Perturbed.png|350]]

---
`Centering`

The dataset is
$$
Y= \begin{bmatrix}
4 & 8 & 12 \\
2 & 7 & 9
\end{bmatrix}
, \quad \mu_{Y} = \begin{bmatrix}
8 \\ 6
\end{bmatrix}
$$

Hence, the `centered dataset` is
$$
Z = \begin{bmatrix}
-4 & 0 & 4 \\
-4 & 1 & 3
\end{bmatrix}
$$
and its `covariance matrix` is
$$
S = S_{Y} = \begin{bmatrix}
16 & 14 \\
14 & 13
\end{bmatrix}
$$

---
`Rotation`

The [[Finding the Orthogonal Matrix|Orthogonal Matrix]] is still the same as in [[Rotating Coordinate System]]
$$
V = \begin{bmatrix}
\frac{4}{5} & -\frac{3}{5} \\
\frac{3}{5} & \frac{4}{5}
\end{bmatrix}
, \quad
W = \begin{bmatrix}
\frac{4}{5} \\ \frac{3}{5}
\end{bmatrix}
, \quad
W^{\perp} = \begin{bmatrix}
-\frac{3}{5} \\ \frac{4}{5}
\end{bmatrix}
$$

Using it, we can rotate the coordinate system to get
$$
X = V^T(Y- \mu_{Y})
= V^TZ
= \begin{bmatrix}
-5.6 & 0.6 & 5 \\
-0.8 & 0.8 & 0
\end{bmatrix}
$$

with `covariance matrix` of
$$
S_{X} = \begin{bmatrix}
28.36 & 2.48 \\
2.48 & 0.64
\end{bmatrix}
$$

---
`Dimension Reduction`

Carrying out similar [[Dimension Reduction]], we get
$$
\hat{X} = \begin{bmatrix}
-5.6 & 0.6 & 5
\end{bmatrix}
, \quad
W\hat{X} = \begin{bmatrix}
-4.48 & 0.48 & 4 \\
-3.36 & 0.36 & 3
\end{bmatrix}
$$
and
$$
\hat{X}^{\perp} 
= \begin{bmatrix}
-0.8 & 0.8 & 0
\end{bmatrix}
, \quad
W^{\perp}
\hat{X}^{\perp}
= \begin{bmatrix}
0.48 & -0.48 & 0 \\
-0.64 & 0.64 & 0
\end{bmatrix}
$$

---
`Reconstruction`

We can get that
$$
\hat{Y} 
= W\hat{X} + \mu_{Y}\cdot1
= \begin{bmatrix}
3.52 & 8.48 & 12 \\
2.64 & 6.36 & 9
\end{bmatrix}
$$

Now, note that
$$
Y - (W^{\perp} \hat{X}^{\perp}) 
= \begin{bmatrix}
4 & 8 & 12 \\
2 & 7 & 9
\end{bmatrix}
- \begin{bmatrix}
0.48 & -0.48 & 0 \\
-0.64 & 0.64 & 0
\end{bmatrix}
= \begin{bmatrix}
3.52 & 8.48 & 12 \\
2.64 & 6.36 & 9
\end{bmatrix}
$$

> This means that $W^{\perp}\hat{X}^{\perp}$ is the `reconstruction error`

---

`Remarks`

`Reconstruction Error`
$$
\begin{align}
\hat{Y}  
&= W\hat{X} + \mu_{Y}1 \\[6pt]
\hat{Y} &= Y- W^{\perp} \hat{X}^{\perp} \\[6pt]
Y - \hat{Y} &= W^{\perp} \hat{X}^{\perp}
\end{align}
$$

`Reconstruction`
Given a new point $\hat{x}^* \in R^{k\times1}$, project back to reconstruct $\hat{y}^* \in R^{d \times 1}$ 

$$
\begin{align}
&\hat{y}^* = W\hat{x}^* + \mu_{Y} \\[6pt]
&\implies\begin{cases}
y - \hat{y}^* = W^{\perp} \hat{x}^{*\perp} \\
\hat{y}^* - \mu_{y} = W\hat{x}^{*}
\end{cases}
\end{align}
$$

![[PCA Pythagorean.png|350]]

`Pythagorean Theorem`
Using Pythagorean Theorem, we can derive that
$$
\begin{align}
&(W\hat{x}^*) \perp (W^{\perp} \hat{x}^{*\perp})  
\\[6pt]
&\implies ||y - \mu_{Y}||^2 = ||y - \hat{y}^*||^2 + ||\hat{y}^* - \mu_{Y} ||^2
\end{align}
$$

---