# Dimension Reduction

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

Note that their `covariance matrices` are
$$
S = S_{Y} 
= S_{Z}
= \begin{bmatrix}
16 & 12 \\
12 & 9
\end{bmatrix}
$$
---
`Rotating`
Recall from [[Rotating Coordinate System]] that using the orthogonal matrix 
$$
V = \begin{bmatrix}
\frac{4}{5} & -\frac{3}{5}  \\
\frac{3}{5} & \frac{4}{5}
\end{bmatrix}
$$
we can get that
$$
A = V^TY
= \begin{bmatrix}
5 & 10 & 15 \\
0 & 0 & 0
\end{bmatrix}
, \quad
\mu_{A} = \begin{bmatrix}
10 \\ 0
\end{bmatrix}
$$

We can then center it to
$$
X = V^TY - \mu_{A}
= \begin{bmatrix}
-5 & 0 & 5 \\
0 & 0 & 0
\end{bmatrix}
, \quad
\mu_{X} = \begin{bmatrix}
0 \\ 0
\end{bmatrix}
$$

Note that their `covariance matrices` are
$$
S_{A} = S_{X} = \begin{bmatrix}
25 & 0  \\
0 & 0
\end{bmatrix}
$$
---
`Reduction`

The variance in the direction of eigenvector $v_{2}$ is $0$.
Hence, we can reduce it to
$$
W  
= \begin{bmatrix}
v_{1}
\end{bmatrix}
= \begin{bmatrix}
\frac{4}{5} \\ \frac{3}{5}
\end{bmatrix}
$$

Using it, we get
$$
\begin{align}
\hat{X} 
&= \begin{bmatrix}
x_{1}
\end{bmatrix} \\[6pt]
&= W^T \ (Y- \mu_{Y}) \\[6pt]
&= \begin{bmatrix}
\frac{4}{5} & \frac{3}{5}
\end{bmatrix}
\begin{bmatrix}
-4 & 0 & 4 \\
-3 & 0 & 3
\end{bmatrix} \\[6pt]
&= \begin{bmatrix}
-5 & 0 & 5
\end{bmatrix}
\end{align}
$$

---

