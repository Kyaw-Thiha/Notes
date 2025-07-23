#cv/projection/orthographic
# Orthographic Projection
Drop the z-component of the 3D vector.
## Partial Homogeneous
$$
\begin{aligned}
\vec{x} 
&= [I_{2 \times 2} | 0].\vec{p} \\[2ex]
	&= \begin{bmatrix}
	1 & 0 & 0 \\ 0 & 1 & 0
	\end{bmatrix}.
	\begin{bmatrix}
	x \\ y \\ z
	\end{bmatrix} \\[2ex]
	
&= \begin{bmatrix}
	x \\ y
	\end{bmatrix}
\end{aligned}
$$

## Homogeneous Representation
Drop the z-component, but keep the w-component.
$$
\begin{aligned}
\tilde{x} 
&= \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0  \\
0 & 0 & 0 & 1
\end{bmatrix} .\tilde{p} \\[2ex]

&= \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0  \\
0 & 0 & 0 & 1
\end{bmatrix} .
\begin{bmatrix}
x \\ y \\ z \\ w
\end{bmatrix} \\[2ex]

&= \begin{bmatrix}
x \\ y \\ 0 \\ w
\end{bmatrix}
\end{aligned}
$$




