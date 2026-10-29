# Trajectory Generation

---
## Task vs Joint Trajectory Prediction
![image|400](https://notes-media.kthiha.com/Robotics-Trajectory-Generation/b62ed6768e78e79d24e9345b1a8a86ea.png)

---
## Trajectory
A [[Robotics Trajectory Generation|trajectory]] is the [[Config Space(C-Space)|robot configuration]]  as a function of time
$$
\theta(t), \; t \in [0, T]
$$

When plotted, it can be visualized as 

![image|500](https://notes-media.kthiha.com/Robotics-Trajectory-Generation/2664de751bd5aa0713b538fb8601794e.png)

---
### Path and Trajectory
$$
\begin{array}{rl}
\textbf{Path} & \theta(s), \; s \in [0,1] \\[0.3em]
& \Big\Downarrow \\[0.3em]
\textbf{Trajectory} & \theta\big(s(t)\big), \; \underbrace{s : [0,T] \to [0,1]}_{\text{Time Scaling}}
\end{array}
$$

---
## Task vs Joint Space
> **Task Space**
> $(+)$ Motion is predictable.
> $(+)$ Better handling of collisions with obstacles.
> $(-)$ [[Inverse Kinematics|Inverse kinematics]] call at every step.
> $(-)$ Actuator motor is not necessarily smooth.

> **Joint Space**
> $(+)$ Fewer [[Inverse Kinematics|inverse kinematics]] calls.
> $(+)$ Actuator motion is smooth and easy to validate.
> $(-)$ Motion may not be collision-free in task-space.

---
## Straight-Line Paths
### Joint Space
In [[#Joint Space|joint space]], its trajectory can be represented as
$$
\theta(s) = \theta_{\text{start}} + s\left(\theta_{\text{end}} - \theta_{\text{start}}\right), \quad s \in [0,1]
$$

The joint angles move linearly with respect to time.

![image|500](https://notes-media.kthiha.com/Robotics-Trajectory-Generation/56935be08bef544b04b811a2b8c0e46e.png)

But its path might not be a straight line.

---
### Task Space
In [[#Task Space|task space]] with $X \in \mathbb{R}^{2}$, its trajectory can be represented as
$$
X(s) = X_{\text{start}} + s\left(X_{\text{end}} - X_{\text{start}}\right), \quad s \in [0,1]
$$

It path motion in the task space is a straight line.

![image|500](https://notes-media.kthiha.com/Robotics-Trajectory-Generation/d95c7c92fb491000dc7d5bc47a7083e0.png)

But the joint angle motion may not be linear with respect to time.

> Reachable task space may not be convex in $X$ coordinates such that coordinates on the straight line may not be reachable.

---
### Task Space Subtraction
Suppose we have $X_{start}$ and $X_{end}$ as 
$$
X_{\text{start}} = \begin{pmatrix}
0.4924 & 0.5868 & -0.6428 & 2.986 \\
-0.4132 & -0.4924 & -0.7660 & -2.506 \\
-0.7660 & 0.6428 & 0 & -0.9508 \\
0 & 0 & 0 & 1
\end{pmatrix}
\quad
X_{\text{end}} = \begin{pmatrix}
-0.9254 & 0.1632 & -0.3420 & -3.419 \\
-0.3368 & 0.0594 & 0.9397 & -1.244 \\
0.1736 & 0.9848 & 0 & 3.4 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

Subtracting them would get us
$$
X_{\text{end}} - X_{\text{start}} = \begin{pmatrix}
-1.4178 & -0.4236 & 0.3008 & -6.4051 \\
0.0764 & 0.5518 & 1.7057 & 1.2611 \\
0.9397 & 0.3420 & 0 & 4.3512 \\
0 & 0 & 0 & 0
\end{pmatrix}
\notin SE(3)
$$

---
#### Log Space
For $X = (R,p) \in SE(3)$, 
$$
\begin{align*}
X_{\text{start,end}} &= X_{\text{start},s} X_{s,\text{end}} \\[1em]
&= X_{s,\text{start}}^{-1} X_{s,\text{end}}
\end{align*}
$$
and converting it to log space get us
$$
\log\left(X_{s,\text{start}}^{-1} X_{s,\text{end}}\right) \in \mathfrak{se}(3)
$$
which is the representation of the twist in $\{ start \}$ taking $X_{start}$ to $X_{end}$ in unit time.

Hence, we get that
$$
X(s) = X_{\text{start}} \exp\Big(\underbrace{\log\left(X_{\text{start}}^{-1} X_{\text{end}}\right) s}_{\color{#2E75B6}\text{Interpolation using twist representation}}\Big), \qquad s \in [0,1]
$$

---
### Decoupled Rotation and Translation
We have that the frame origin follows a straight line.

![image|400](https://notes-media.kthiha.com/Robotics-Trajectory-Generation/08e4b5f69a812f5feb75cd4bd9606a85.png)

By decoupling the rotation and translation, we can get
$$
\begin{align*}
p(s) &= p_{\text{start}} + s\left(p_{\text{end}} - p_{\text{start}}\right), \\[1em]
R(s) &= R_{\text{start}} \exp\left(\log\left(R_{\text{start}}^T R_{\text{end}}\right) s\right), \quad s \in [0,1]
\end{align*}
$$

---
## Time Scaling
We can use [[Time Scaling Trajectory|time scaling]] to convert a [[Robotics Trajectory Generation|path]] into a [[Robotics Trajectory Generation|trajectory]] 
$$
\begin{array}{rl}
\textbf{Path} & \theta(s), \; s \in [0,1] \\[0.3em]
& \Big\Downarrow \\[0.3em]
\textbf{Trajectory} & \theta\big(s(t)\big), \; \underbrace{s : [0,T] \to [0,1]}_{\text{Time Scaling}}
\end{array}
$$
We can [[Time Scaling Trajectory|time-scale]] [[Robotics Trajectory Generation|trajectories]] using one of
- [[Time Scaling Trajectory#Third-Order Polynomial Time Scaling|Third-Order Polynomial Time Scaling]]
- [[Time Scaling Trajectory#Fifth-Order Polynomial Time Scaling|Fifth-Order Polynomial Time Scaling]]
- [[Time Scaling Trajectory#Trapezoidal Motion Profile|Trapezoidal Motion Profile]]
- [[Time Scaling Trajectory#S-Curve Time Scaling|S-Curve Time Scaling]]

[[Time Scaling Trajectory|Read More]]

---
