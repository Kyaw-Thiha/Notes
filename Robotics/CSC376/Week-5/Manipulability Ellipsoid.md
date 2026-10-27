# Manipulability Ellipsoid
A [[Manipulability Ellipsoid|manipulability ellipsoid]] visualizes how easily robot manipulator can move its end-effector in different directions at a given joint configuration.

![image|400](https://notes-media.kthiha.com/Manipulability-Ellipsoid/6b180564db7ee435bb92387e61a6286f.png)

---
## Computing the Manipulability Ellipsoid
Given that every joint can move with same maximum speed, we want to know what tip velocities are possible.
$$
\begin{array}{lll}
v = J(\theta)\dot{\theta} & & \\[1em]
\dot{\theta}^T\dot{\theta} = 1 & & \text{Assume all joint velocity vectors have unit norm.} \\[1em]
v^T J^{-T} J^{-1} v = 1 & & \text{Substitute } \dot{\theta} = J^{-1}v \\[1em]
\boxed{v^T (JJ^T)^{-1} v = 1} & & \text{Since } (JJ^T)^{-1} = J^{-T}J^{-1}
\end{array}
$$
where $v^T (JJ^T)^{-1} v = 1$ is the standard equation of an ellipsoid.

![image|300](https://notes-media.kthiha.com/Manipulability-Ellipsoid/e8e2789e2d2a18d7c1baa5f34f9d3410.png)

where
$$
\begin{array}{lll}
v^T A^{-1} v = 1 & & \\[1em]
A = JJ^T \in \mathbb{R}^{m \times m} & & \text{(symmetric, positive definite)} \\[1em]
A = \lambda_1, \dots, \lambda_m & & \text{Eigenvalues} \\[1em]
A = v_1, \dots, v_m & & \text{Eigenvectors}
\end{array}
$$

---
## Near Kinematic Singularities
![image|500](https://notes-media.kthiha.com/Manipulability-Ellipsoid/bdb20e7161a799402313e8ff08afbddb.png)

---
## Manipulability Measures
### Ratio of Longest & Shortest Axes
Given the [[Manipulability Ellipsoid|manipulability ellipsoid]], 

![image|300](https://notes-media.kthiha.com/Manipulability-Ellipsoid/e8e2789e2d2a18d7c1baa5f34f9d3410.png)

We can get a [[#Manipulability Measures|manipulability measure]] using the ratio of longest and shortest axes
$$
\begin{align*}
\mu_1(A) &= \frac{\sqrt{\lambda_{\max}(A)}}{\sqrt{\lambda_{\min}(A)}} \qquad A = JJ^T \in \mathbb{R}^{m \times m} \\[1em]
&= \sqrt{\frac{\lambda_{\max}(A)}{\lambda_{\min}(A)}} \ge 1 \\
&\phantom{=\sqrt{\frac{\lambda_{\max}(A)}{\lambda_{\min}(A)}}\;} = 1: \text{ isotropic}
\end{align*}
$$

where the smaller values closer to $1$ are better for manipulability.

---
### Condition Number of A
Given the [[Manipulability Ellipsoid|manipulability ellipsoid]], 

![image|300](https://notes-media.kthiha.com/Manipulability-Ellipsoid/e8e2789e2d2a18d7c1baa5f34f9d3410.png)

We can get a [[#Manipulability Measures|manipulability measure]] using the ratio of longest and shortest axes
$$
\mu_2(A) = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)} \ge 1 \qquad A = JJ^T \in \mathbb{R}^{m \times m}
$$
where the smaller values closer to $1$ are better for manipulability.

---
### Proportionality to Volume of Ellipsoid
Given the [[Manipulability Ellipsoid|manipulability ellipsoid]], 

![image|300](https://notes-media.kthiha.com/Manipulability-Ellipsoid/e8e2789e2d2a18d7c1baa5f34f9d3410.png)

We can get a [[#Manipulability Measures|manipulability measure]] using the proportionality to volume of ellipsoid.
$$
\begin{align*}
\mu_3(A) &= \sqrt{\lambda_1 \lambda_2 \cdots} \qquad A = JJ^T \in \mathbb{R}^{m \times m} \\[1em]
&= \sqrt{\det(A)}
\end{align*}
$$
where the larger values are better.

---
## Angular and Linear Velocity Manipulability Ellipsoids
Given the [[Robot Jacobian|body Jacobian]] of
$$
J_b(\theta) = \begin{bmatrix} J_{b\omega}(\theta) \\ J_{bv}(\theta) \end{bmatrix} \in \mathbb{R}^{6 \times n}
$$
we can get 
$$
\begin{array}{l}
J_{b\omega}(\theta) \in \mathbb{R}^{3 \times n}: \text{ angular velocity ellipsoid} \\[1em]
J_{bv}(\theta) \in \mathbb{R}^{3 \times n}: \text{ linear velocity ellipsoid}
\end{array}
$$

---