# Centering Coordinate System 

`Centering the dataset`

![[PCA (Centering).png|300]]

Note that 
$$
y_{1} = \begin{bmatrix}
4 \\ 3
\end{bmatrix}
, \ 
y_{2} = \begin{bmatrix}
8 \\ 6
\end{bmatrix}
, \
y_{3} = \begin{bmatrix}
12 \\ 9
\end{bmatrix}
$$

Hence, we get that 
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
`Mean`

Now, lets compute the `mean` of the dataset.
$$
\mu_{Y} 
= \frac{1}{N} \sum^{N}_{i=1} y_{i}
= \frac{1}{3} \begin{bmatrix}
4 + 8 + 12 \\
3 + 6 + 9
\end{bmatrix}
= \begin{bmatrix}
8 \\ 6
\end{bmatrix}
$$

Next, we will be using the `mean` to center our dataset.
$$
\begin{align}
Z  
&= Y - \mu_{Y}.1 \\[6pt]
&= \begin{bmatrix}
-4 & 0 & 4 \\
-3 & 0 & 3
\end{bmatrix}
\end{align}
$$

Analyzing the `mean` of this centered dataset, we get
$$
\mu_{Z} 
= \frac{1}{N} \sum^N_{i=1} z_{i}
= \begin{bmatrix}
0 \\ 0
\end{bmatrix}
$$

---
`Variance`

Analyzing the `variance` of the original dataset, we get
$$
\begin{align}
S_{Y} 
&= \frac{1}{N-1} (Y - \mu_{Y}1) (Y - \mu_{Y}1)^T \\[6pt]
&= \frac{1}{N-1} Z Z^T
\end{align}
$$

Using this to find the `variance` of the centered dataset,
$$
\begin{align}
S_{Z} 
&= \frac{1}{2} 
\begin{bmatrix}
-4 & 0 & 4 \\
-3 & 0 & 3
\end{bmatrix}
\begin{bmatrix}
-4 & -3 \\
0 & 0 \\
4 & 3
\end{bmatrix} \\[6pt]
&= \begin{bmatrix}
16 & 12 \\
12 & 9
\end{bmatrix} \\[6pt]
&= \begin{bmatrix}
\sigma_{1}^2 & \sigma_{1} \sigma_{2}  \\
\sigma_{2} \sigma_{1} & \sigma_{2}^2
\end{bmatrix}
\end{align}
$$

---
`Remarks`

- `Covariance`
  Since centering the data only shifts the coordinate system without rotation, it does not change the `covariance matrix`

- `Variance`
  The `variance` of the data in one direction in the coordinate system is also the `variance` of all data projected onto that direction.
$$
\begin{align}
Var(e_{i}^T \ Y)
&= \frac{1}{N-1} || e_{i}^TY  - e_{i}^T \mu_{Y}.1||^2_{2} \\[6pt]
&= \frac{1}{N-1}  
\left( 
e_{i}^T \ (Y - \mu_{Y}1)(Y - \mu_{Y}1)^T \ e_{i}  
\right) \\[6pt]
&= e_{i}^T \ S \ e_{i} \\[6pt]
&= \sigma_{i}^2
\end{align}
$$

- `Diagonals`
  The `diagonals` of the `Covariance Matrix` contain all the `variances` of data in each direction of the coordinate system.

- `Trace`
  The sum of the diagonals $Tr(S)$ is the `variance` of the dataset.
  
---


