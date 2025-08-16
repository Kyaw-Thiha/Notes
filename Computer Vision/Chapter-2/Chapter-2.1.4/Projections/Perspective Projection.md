#cv/projection/perspective
# Perspective Projection

## Non-homogeneous
$$
\tilde{x} = P_{z}(\vec{p})
= \begin{bmatrix}
\frac{x}{z} \\ \frac{y}{z} \\ 1
\end{bmatrix}
$$
## Homogeneous
Simply dropping the $w$ component.
This makes sense as after projection, it is not possible to recover distance of the 3D point from the image.
$$
\begin{aligned}
\tilde{x} 
&= \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}. \tilde{p} \\[2ex]

&= \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}. 
\begin{bmatrix}
x \\ y \\ z \\ w
\end{bmatrix} \\[2ex]

&= \begin{bmatrix}
x \\ y \\ z
\end{bmatrix}
\end{aligned}
$$
