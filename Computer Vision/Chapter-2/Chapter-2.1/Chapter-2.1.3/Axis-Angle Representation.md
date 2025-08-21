# Exponential Twist (Axis/Angle)
#math #cv/transformations/3d/rotation/axis-angle #rodrigues_formula 

![Axis-Angle Representation Diagram](Axis_Angle_Representation.svg)

Representing 3D rigid body rotation with
- Axis of Rotation (unit vector $u \in \mathbb{R}^3$)
- Angle $\theta \in \mathbb{R}^3$  


## Formula
$$
(\text{axis}, \text{angle}) = (
\begin{bmatrix}
e_{x} \\ e_{y} \\ e_{z}
\end{bmatrix}, \theta
)
$$

## Rodrigues Formula
A formula that is used to efficiently calculate the rotation of a vector based on Axis-Angle representation.
$$
R(\hat{n}, \theta) = 
\text{I} + 
\sin(\theta).[\hat{n}]_{x} +
(1-\cos(\theta))[\hat{n}]^2_{x}
$$
where $[n]_{x}$ is the skew-symmetric matrix
$$
[n]_{x} = 
\begin{bmatrix}
0 & -u_{z} & u_{y} \\
u_{z} & 0 & -u_{x} \\
-u_{y} & u_{x} & 0
\end{bmatrix}
$$

$R(\hat{n}, \theta)$ can then be applied to $\vec{v}$ to get $\vec{v'}$
$$
\vec{v'} = R(\hat{n}, \theta).\vec{v}
$$

## Example
Suppose we wish to rotate in $\vec{u} = [0, 0, 1]^T$ axis for $\theta=\frac{\pi}{2}$
Then,
$$
[u]_{x} = 
\begin{bmatrix}
0 & -1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

$$
[u]_{x}^2 = 
\begin{bmatrix}
-1 & 0 & 0 \\
0 & -1 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$
Using Rodrigues formula, we get
$$
\begin{aligned}
	R(\hat{n}, \theta) 
	&= \text{I} + 
	\sin\left( \frac{\pi}{2} \right).[\hat{n}]_{x} +
	\left( 1-\cos\left( \frac{\pi}{2} \right) \right)
	[\hat{n}]^2_{x} \\[3ex]

	&= I + [u]_{x} + [u]_{x}^2 \\[3ex]
	&= \begin{bmatrix}
			0 & -1 & 0 \\
			1 & 0 & 0  \\
			0 & 0 & 1
		\end{bmatrix}
\end{aligned}
$$
which is the Rotation matrix of $\frac{\pi}{2}$ about z-axis, as wanted!

## Deriving of Rodrigues Forumula

## Read also
- [Axis-Angle Wikipedia](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Example)
- [Rodrigues' Formula Wikipedia](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
