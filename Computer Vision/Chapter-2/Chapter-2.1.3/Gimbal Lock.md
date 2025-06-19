# Gimbal Lock
#math #robotics #cv/transformations/3d/rotation  

When using Euler angles to describe 3D rotation, we have to describe in composition of $R_{x}$, $R_{y}$, $R_{z}$
The problem arises when at some points, the rotation can drop from 3 degrees of freedom to 2 or even 1.

## Graphical Example
![Gimbal Lock GIF](./attachments/Gimbal_Lock_Plane.gif)
When the pitch (green) and yaw (magenta) gimbals become aligned, changes to roll (blue) and yaw(magenta) apply the same rotation to the airplane.

---
## Example

Consider the following example
$$
\begin{aligned}
R &= R_{x}.R_{y}.R_{z} \\
&= 
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos(\alpha) & -\sin(\alpha) \\ 
0 & \sin(\alpha) & \cos(\alpha) \\
\end{bmatrix}
.
\begin{bmatrix}
\cos(\beta) & 0 & \sin(\beta) \\
0 & 1 & 0 \\ 
-\sin(\beta) & 0 & \cos(\beta) \\
\end{bmatrix}
.
\begin{bmatrix}
\cos(\gamma) & -\sin(\gamma) & 0 \\
\sin(\gamma) & \cos(\gamma) & 0 \\ 
0 & 0 & 1 \\
\end{bmatrix}

\end{aligned}
$$

Consider $\beta = \frac{\pi}{2}$
$$

\begin{aligned}

R &= 
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos(\alpha) & -\sin(\alpha) \\ 
0 & \sin(\alpha) & \cos(\alpha) \\
\end{bmatrix}
.
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\ 
-1 & 0 & 0 \\
\end{bmatrix}
.
\begin{bmatrix}
\cos(\gamma) & -\sin(\gamma) & 0 \\
\sin(\gamma) & \cos(\gamma) & 0 \\ 
0 & 0 & 1 \\
\end{bmatrix} 

\\[3ex]

&=
\begin{bmatrix}
1 & 0 & 0 \\
0 & \sin(\alpha) & \cos(\alpha) \\ 
0 & -\cos(\alpha) & \sin(\alpha) \\
\end{bmatrix}
.
\begin{bmatrix}
\cos(\gamma) & -\sin(\gamma) & 0 \\
\sin(\gamma) & \cos(\gamma) & 0 \\ 
0 & 0 & 1 \\
\end{bmatrix}

\\[3ex]

&=
\begin{bmatrix}
0 & 0 & 1 \\

\sin(\alpha).\cos(\gamma) + \cos(\alpha).\sin(\gamma) &  
-\sin(\alpha).\sin(\gamma) + \cos(\alpha).\cos(\gamma) & 
0 \\ 

-\cos(\alpha).\cos(\gamma) + \sin(\alpha).\sin(\gamma) &  
-\cos(\alpha).\sin(\gamma) + \sin(\alpha).\cos(\gamma) &
0 \\
\end{bmatrix}

\\[3ex]

&= 
\begin{bmatrix}
0 & 0 & 1 \\
\sin(\alpha + \gamma) & \cos(\alpha + \gamma) & 0 \\ 
-\cos(\alpha + \gamma) & \sin(\alpha + \gamma) & 0 \\
\end{bmatrix}

\end{aligned}

$$

Choosing $\theta = \alpha + \gamma$, we get
$$
\begin{aligned}
R &= 

\begin{bmatrix}
0 & 0 & 1 \\
\sin(\theta) & \cos(\theta) & 0 \\ 
-\cos(\theta) & \sin(\theta) & 0 \\
\end{bmatrix}

\\[3ex]

&= R_{z}
\end{aligned}
$$
This means that at $\beta = \frac{\pi}{2}$, our Euler rotation has been locked to the z-axis rotation only.

---
## Mathematical Explanation
In group theory, the 3D rotations are considered as [SO(3)](https://en.wikipedia.org/wiki/3D_rotation_group)

The map from Euler angles to rotations (topologically from $T^3$ torus to real projective space $RP^3$) is not a [local homeomorphism](https://en.wikipedia.org/wiki/Local_homeomorphism) at every points.

Therefore, there are specific points at which the rank of the map(degrees of freedom) drop below 3.

---
## Ways to overcome it
- Use Rodrigues' formula
- Use quaternions representation


> **Fun Fact:** Gimbal lock happened on the Appollo-11 mission (The one that humans landed on the Moon)

---
## Read Also
- [Wikipedia](https://en.wikipedia.org/wiki/Gimbal_lock)
- [PDF Explanation](https://math.umd.edu/~immortal/MATH431/book/ch_gimballock.pdf)
- [Video Example](https://www.youtube.com/watch?v=z3dDsz4f20A&t=154s)
- [Local Homeomorphism](https://en.wikipedia.org/wiki/Local_homeomorphism)
- [Homeomorphism](https://en.wikipedia.org/wiki/Homeomorphism)
