#cv/projection/scaled-orthographic
# Scaled Orthographic Projection
In practice, world coordinates maybe measured in meters, and thus, need to be rescaled onto image sensors, usually in millimeters then pixels.

This is essentially [[Orthographic Projection]], but with a scale.

- Projected orthogonally to local reference (fronto-parallel) plane 
- Next, projection onto final image plane
## Partial Homogeneous
$$
\begin{aligned}
\vec{x} 
&= [s.I_{2 \times 2} | 0].\vec{p} \\[2ex]
	&= \begin{bmatrix}
	s & 0 & 0 \\ 0 & s & 0
	\end{bmatrix}.
	\begin{bmatrix}
	x \\ y \\ z
	\end{bmatrix} \\[2ex]
	
&= \begin{bmatrix}
	s.x \\ s.y
	\end{bmatrix}
\end{aligned}
$$

## Homogeneous Representation
Drop the z-component, but keep the w-component.
$$
\begin{aligned}
\tilde{x} 
&= \begin{bmatrix}
s & 0 & 0 & 0 \\
0 & s & 0 & 0  \\
0 & 0 & 0 & 1
\end{bmatrix} .\tilde{p} \\[2ex]

&= \begin{bmatrix}
s & 0 & 0 & 0 \\
0 & s & 0 & 0  \\
0 & 0 & 0 & 1
\end{bmatrix} .
\begin{bmatrix}
x \\ y \\ z \\ w
\end{bmatrix} \\[2ex]

&= \begin{bmatrix}
s.x \\ s.y  \\ w
\end{bmatrix}
\end{aligned}
$$



