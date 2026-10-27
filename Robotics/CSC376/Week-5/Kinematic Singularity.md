# Singularity
[[Kinematic Singularity|Singularity]] ([[Kinematic Singularity|kinematic singularity]]) is a robot posture at which the robot's end-effector loses the ability to move instantaneously in one or more directions. 

![300](https://ars.els-cdn.com/content/image/1-s2.0-S0016003220307894-gr6.jpg)

---
## Mathematical Definition
> The [[Robot Jacobian|Jacobian]] $J(\theta) \in \mathbb{R}^{6 \times n}$ maps joint velocities to end effector velocity (linear + angular so $6$ rows). With $n$ joints, its a $6 \times n$ matrix.
> 
> Its rank is no. of independent directions the end-effector can instantaneously move in.

$$
\begin{array}{ll}
 & \operatorname{rank} J(\theta) \le \min(6, n) \\[1em]
\textbf{full rank} & \operatorname{rank} J(\theta) = \min(6, n) \\[1em]
\textbf{singular} & \text{at } \theta^* \text{ if } \operatorname{rank} J(\theta^*) < \max_{\theta} \operatorname{rank} J(\theta)
\end{array}
$$

The $\text{rank}$ $J(\theta)$ $\leq$ $\min(6, n)$ since a matrix cannot have more independent rows/columns than what it actually has. So it has full rank if $\text{rank} \ J(\theta) = \min(6,n)$. 

 A pose $\theta^{*}$ is [[Kinematic Singularity|singular]] is $\text{rank } J(\theta^{*})$ is less than maximum rank $J$ can achieve anywhere else.

---
## Categorizing Jacobian by the Number of Joints

> **Case-1**: $n < 6$
> $J$ is tall. The robot is kinematically deficient.

> **Case-2**: $n = 6$
> $J$ is square. The robot is general 6D rigid-body motion.

> **Case-3**: $n > 6$
> $J$ is fat. The robot is kinematically redundant.

---
## Example
### Zero-Config
Consider the following 3R robot.

![image|400](https://notes-media.kthiha.com/Kinematic-Singularity/31f36aafe5f275c0fa9de778a67ebe6d.png)

It thus have the following [[DH Parameter (Denavit-Hartenberg)|DH table]].
$$
\begin{array}{|c||c|c|}
\hline
\mathcal{S}_i & \omega_i & v_i \\
\hline\hline
1 & (0,0,1) & (0,0,0) \\
2 & (0,0,1) & (0,-2,0) \\
3 & (0,0,1) & (0,-4,0) \\
\hline
\end{array}
$$

Its [[Robot Jacobian|Jacobian]] is $J_{S}(\theta) \in \mathbb{R}^{6 \times 3}$ and its rank is
$$
\begin{array}{l}
\text{full rank } 3 \\[]
\text{singular configuration for } \operatorname{rank}\big(J_s(\theta)\big) < 3
\end{array}
$$
---
### Config-1
Consider the robot in the configuration as shown below.

![image|200](https://notes-media.kthiha.com/Kinematic-Singularity/cf52817c52164d07f3d4b67bc1db212e.png)

We have the joint angles of 
$$
\theta = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \theta_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 45^\circ \\ 30^\circ \end{bmatrix}
$$
Evaluating the [[Robot Jacobian|Jacobian]], we get
$$
J_s(\theta) = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
1 & 1 & 1 \\
0 & 0 & 1.414 \\
0 & -2 & -3.414 \\
0 & 0 & 0
\end{bmatrix}

\qquad

\operatorname{rank}(J_s) = 3
$$

---
### Config-2
Consider the robot in the configuration as shown below.

![image|300](https://notes-media.kthiha.com/Kinematic-Singularity/3190241ca95b6dac40633335013614a0.png)

We have the joint angles of 
$$
\theta = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \theta_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
$$

Evaluating the [[Robot Jacobian|Jacobian]], we get
$$
\begin{align*}
J_s(\theta) &= \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
1 & 1 & 1 \\
0 & 0 & 0 \\
0 & -2 & -4 \\
0 & 0 & 0
\end{bmatrix} \\[1em]
\operatorname{rank}(J_s) &= 2 < 3 \;\rightarrow\; \text{robot is in a singular configuration}
\end{align*}
$$
Any [[Velocity Kinematics|joint velocity vector]] $\dot{\theta}$ only generates vertical linear velocities.

---
## Common Kinematic Singularities
### Two Collinear Revolute Joints
Consider the following robot configuration with two collinear revolute joints.

![image|100](https://notes-media.kthiha.com/Kinematic-Singularity/0920913a27268f483a765d7f2f8b619c.png)

It has the [[Robot Jacobian|Jacobian]] of 
$$
J_{s1}(\theta) = \begin{bmatrix} \omega_{s1} \\ -\omega_{s1} \times q_1 \end{bmatrix} \qquad J_{s2}(\theta) = \begin{bmatrix} \omega_{s2} \\ -\omega_{s2} \times q_2 \end{bmatrix}
$$
Given the configuration, it has the constraints of
$$
\begin{align*}
\omega_{s1} &= \pm\,\omega_{s2} \\[1em]
\omega_{si} \times (q_1 - q_2) &= 0 \quad \text{for } i = 1, 2
\end{align*}
$$
Thus, we can conclude that 
$$
J_{s1} = J_{s2} \;\Longrightarrow\; \text{not linearly independent}
$$

---
### Three Coplanar and Parallel Revolute Joint Axis
Consider the following robot configuration with three coplanar and parallel revolute joint axes.

![image|250](https://notes-media.kthiha.com/Kinematic-Singularity/39da051f657600f25ecfe4edbb41755f.png)

It has the [[Robot Jacobian|Jacobian]] of 
$$
J_s(\theta) = \begin{bmatrix}
\omega_{s1} & \omega_{s1} & \omega_{s1} & \cdots \\
0 & -\omega_{s1} \times q_2 & -\omega_{s1} \times q_3 & \cdots
\end{bmatrix}
$$
Given the configuration, it has the constraints of
$$
\begin{array}{r}
q_2 \text{ and } q_3 \text{ points on the same unit axis} \\[1em]
\Longrightarrow J_{s1}, J_{s2}, J_{s3} \text{ are not linearly independent}
\end{array}
$$

---
### Four Revolute Joint Axes Intersecting in Common Point
Consider the following robot configuration with four revolute joint axes intersecting in a common point.

![image|300](https://notes-media.kthiha.com/Kinematic-Singularity/da216cc2050894f7f21a083415eebf17.png)

It has a [[Robot Jacobian|Jacobian]] of
$$
J_s(\theta) = \begin{bmatrix}
\omega_{s1} & \omega_{s2} & \omega_{s3} & \omega_{s4} & \cdots \\
0 & 0 & 0 & 0 & \cdots
\end{bmatrix}
$$
Given the configuration, it has the constraints of
$$
\begin{array}{c}
q_1 = \cdots = q_4 = 0 \\[1em]
\Longrightarrow J_{s1}, J_{s2}, J_{s3}, J_{s4} \text{ are not linearly independent}
\end{array}
$$

Eg: Wrist center of an elbow-type robot is directly above its shoulder.

---
### Six Revolute Joint Intersecting in a Common Line
Consider the following six revolute joint intersecting in a common line.

![image|100](https://notes-media.kthiha.com/Kinematic-Singularity/0f7483c95bc1a9d02a56a3d184018e12.png)

We thus have the constraints of
$$
\text{Each } q_i = (0, \; 0, \; q_{iz})
$$
Hence, it has the [[Robot Jacobian|Jacobian]] of
$$
\begin{align*}
v_{si} &= -\omega_{si} \times q_i = (\omega_{siy}\, q_{iz}, \; -\omega_{six}\, q_{iz}, \; 0) \\[1em]
J_s(\theta) &= \begin{bmatrix}
\omega_{s1x} & \omega_{s2x} & \omega_{s3x} & \omega_{s4x} & \omega_{s5x} & \omega_{s6x} \\
\omega_{s1y} & \omega_{s2y} & \omega_{s3y} & \omega_{s4y} & \omega_{s5y} & \omega_{s6y} \\
\omega_{s1z} & \omega_{s2z} & \omega_{s3z} & \omega_{s4z} & \omega_{s5z} & \omega_{s6z} \\
\omega_{s1y}q_{1z} & \omega_{s2y}q_{2z} & \omega_{s3y}q_{3z} & \omega_{s4y}q_{4z} & \omega_{s5y}q_{5z} & \omega_{s6y}q_{6z} \\
-\omega_{s1x}q_{1z} & -\omega_{s2x}q_{2z} & -\omega_{s3x}q_{3z} & -\omega_{s4x}q_{4z} & -\omega_{s5x}q_{5z} & -\omega_{s6x}q_{6z} \\
0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
\end{align*}
$$

Hence, we can conclude that
$$
\Longrightarrow J_{si} \text{ are not linearly independent}
$$

---