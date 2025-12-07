# Rotating Coordinate System

`Rotating the Dataset`

![[PCA (Rotation).png|350]]

We have 
$$
Y =
\begin{bmatrix}
| & | & | \\
y_{1} & y_{2} & y_{3} \\
| & | & |
\end{bmatrix}
=
\begin{bmatrix}
4 & 8 & 12 \\
3 & 6 & 9
\end{bmatrix}_{2 \times 3}
$$

---
`Finding the Orthogonal Matrix`

From [[Centering Coordinate System]], recall that we get the covariance matrix of 
$$
S = \begin{bmatrix}
16 & 12 \\
12 & 9
\end{bmatrix}
= \begin{bmatrix}
\sigma_{1}^2 & \sigma_{1}\sigma_{2} \\
\sigma_{2}\sigma_{1} & \sigma_{2}^2
\end{bmatrix}
$$

From [[Finding the Orthogonal Matrix]], we get
$$
V = \begin{bmatrix}
v_{1} & v_{2}
\end{bmatrix}
= \begin{bmatrix}
\frac{4}{5} & -\frac{3}{5}  \\
\frac{3}{5} & \frac{4}{5}
\end{bmatrix}
$$

---
`Rotating the Matrix`

Using the `Orthogonal Matrix` to rotate the dataset,
$$
A = V^TY
= \begin{bmatrix}
5 & 10 & 15 \\
0 & 0 & 0
\end{bmatrix}
$$

Computing the `mean` of the rotated dataset,
$$
\mu_{A}
= \frac{1}{N} \sum^N_{i=1} a_{i}
= \begin{bmatrix}
10 \\ 0
\end{bmatrix}
$$

Using the `mean` to center the dataset,
$$
X = V^TY - \mu_{A} \cdot 1 
= \begin{bmatrix}
-5 & 0 & 5 \\
0 & 0 & 0
\end{bmatrix}
$$

---
`Covariance`
Analyzing the `Covariance Matrix` of the rotated dataset,

$$
\begin{align}
S_{A}  
&= \frac{1}{N-1}  
(V^TY - \mu_{A} \cdot 1)(V^TY - \mu_{A}\cdot 1)^T  
\\[6pt]
&= \frac{1}{N-1} XX^T
\end{align}
$$

Using that to analyze the `Covariance Matrix` of the rotated-centered dataset,
$$
\begin{align}
S_{X}
&= \frac{1}{2}
\begin{bmatrix}
-5 & 0 & 5 \\
0 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
-5 & 0 \\
0 & 0 \\
5 & 0
\end{bmatrix} \\[6pt]
&= \begin{bmatrix}
25 & 0 \\
0 & 0
\end{bmatrix} \\[6pt]
&= \begin{bmatrix}
\sigma_{1}^2 & \sigma_{1} \sigma_{2} \\
\sigma_{2} \sigma_{1} & \sigma_{2}^2
\end{bmatrix}
\end{align}
$$

---

`Remarks`

`Mean`
Note that the relationship between $\mu_{A}$ and $\mu_{Y}$ is
$$
\begin{align}
\mu_{A} 
&= E\ [A] \\[6pt]
&= E(V^TY) \\[6pt]
&= V^T E(Y) \\[6pt]
&= V^T \mu_{Y}
\end{align}
$$

`Covariance`
Using the mean relationship, we can note that the relationship between $S_{A}$ and $S_{Y}$ is
$$
\begin{align}
S_{A}
&= \frac{1}{N-1} 
(V^TY - \mu_{A}1) (V^TY - \mu_{1} 1)^T \\[6pt]
&= \frac{1}{N-1} (V^TY - V^T\mu_{Y}1)
(V^TY - V^T \mu_{A}1)^T \\[6pt]
&= V^T \frac{1}{N-1} (Y - \mu_{Y}1) (Y - \mu_{Y}1)^T \ V \\[6pt]
&= V^T S_{Y} V
\end{align}
$$

`Trace`
Note that the relationship of traces is
$$
\begin{align}
Tr(S_{A})
&= Tr(V^T S_{Y} V) \\[6pt]
&= Tr(S_{Y} \ VV^T) & \text{by cyclic property}  
\\[6pt]
&= Tr(S_{Y} \ I_{d}) \\[6pt]
&= Tr(S_{Y})
\end{align}
$$

Hence, $Tr(S_{A}) = Tr(S_{Y})$.

Since `trace` of the covariance matrix denotes the variance of the dataset, we can deduce that the variance of the dataset does not change.

---
