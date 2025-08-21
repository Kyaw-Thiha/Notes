#cv/projection/para-perspective 
# Para-Perspective Projection
This is a approximation between [[Scaled Orthographic Projection]] and [[Perspective Projection]].

- Projected parallel (not orthogonally)to local reference (fronto-parallel) plane 
- Next, projection onto final image plane

Its main benefits are
- Simpler maths than perspective projection
- Linear Transformation $\implies$ great for computation
- Works well when object is far from camera

It main drawbacks are
- Bad when depth variation is significant.
- Loses accurate depth information.
- Not truly realistic

## Mathematical Explanation
Consider a 3D point ($X, Y, Z$), with a camera facing Z-axis.

In **Orthographic projection**, depth effect is ignored.
$$
\begin{aligned}
x = X \\
y = Y
\end{aligned}
$$

In **Perspective projection**, 
$$
\begin{aligned}
x &= \frac{fX}{Z} \\
y &= \frac{fY}{Z}
\end{aligned}
$$
where $Z$ is the specific depth

In **Para-Perspective projection**,
$$
\begin{aligned}
x &= \frac{fX}{Z_{0}} \\
y &= \frac{fY}{Z_{0}}
\end{aligned}
$$
where $Z_{0}$ is the average depth


## Homogeneous
The combination of the 2 projections can be represented in an affine matrix.
$$
\begin{aligned}
\tilde{x} 
&= \begin{bmatrix} 
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\ 
0 & 0 & 0 & 1
\end{bmatrix}.\tilde{p} \\[2ex]

&= \begin{bmatrix} 
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\ 
0 & 0 & 0 & 1
\end{bmatrix}.
\begin{bmatrix}
x \\ y \\ z \\ w
\end{bmatrix} \\[2ex]

&= \begin{bmatrix}
a_{00}.x + a_{01}.y + a_{02}.z + a_{03}.w \\
a_{10}.x + a_{11}.y + a_{12}.z + a_{13}.w \\
w
\end{bmatrix}
\end{aligned}
$$

