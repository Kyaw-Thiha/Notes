# Product of Exponentials
[[Product of Exponentials|Product of exponentials]] can describe the [[Forward Kinematics|forward kinematics]] of manipulators by multiplying [[Exponential Coordinates|matrix exponentials]] representing each [[Screw Axis|joint screw motions]].

![image|500](https://notes-media.kthiha.com/Product-of-Exponentials/1d5d8747e9157e6d9525c34c30abb224.png)

---
## Illustration
![image|300](https://notes-media.kthiha.com/Product-of-Exponentials/0bdeef30dbe6a6b3197e44b348485bb5.png)

---
## Product of Exponential Steps
### Space Coordinate System
> $(1)$ **Define M**
> Let $M$ be the transformation matrix of the end-effector frame $\{ b \}$ when $\theta=0$.

> $(2)$ **Screw Axis**
> Find the [[Screw Axis|screw axis]] $S_{1}, \ \dots, \ S_{n}$ for each of the $n$ joint axes in the $\{ s \}$-frame when $\theta=0$.

> $(3)$ **Forward Kinematics: Evaluate PoE**
> Given $\theta,$ evaluate the [[Product of Exponentials|product of exponentials formula]] in space frame to determine $T_{sb}(\theta)$.

$$
T_{sb}(\theta) = e^{[S_{1}]\theta_{1}} \ e^{[S_{2}]\theta_{2}} \ \dots \ 
e^{[S_{n}]\theta_{n}} \ M
$$
---
### Body Coordinate System
> $(1)$ **Define M**
> Let $M$ be the transformation matrix of the end-effector frame $\{ b \}$ when $\theta=0$.

> $(2)$ **Screw Axis**
> Find the [[Screw Axis|screw axis]] $\mathcal{B}_{1}, \ \dots, \ \mathcal{B}_{n}$ for each of the $n$ joint axes in the $\{ b \}$-frame when $\theta=0$.

> $(3)$ **Forward Kinematics: Evaluate PoE**
> Given $\theta,$ evaluate the [[Product of Exponentials|product of exponentials formula]] in body frame to determine $T_{sb}(\theta)$.

$$
T_{sb}(\theta) = M \ e^{[\mathcal{B}_{1}]\theta_{1}} \ 
e^{[\mathcal{B}_{2}]\theta_{2}} \ \dots \ 
e^{[\mathcal{B}_{n}]\theta_{n}} 
$$

---
## Revolute-Prismatic-Revolute(RPR) Robot Example
### Space Coordinate
#### Step-1: Determine the transformation $M$ matrix.
Given the following zero position of the robot

![image|400](https://notes-media.kthiha.com/Product-of-Exponentials/2ed91afc0c46a086c07a52a104a01a61.png)

We get the following matrices:
$$
\theta = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \theta_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
\qquad
M = \begin{bmatrix}
1 & 0 & 0 & 4 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

---
#### Step-2: Find screw axis $S$ in $\{ s \}$-frame
##### Finding $\theta_{3}$
For $\theta_{3}$, we have

![image|300](https://notes-media.kthiha.com/Product-of-Exponentials/7c3c289ccb46d02b45ec7cdf5518a592.png)

Given that its direction of rotation is $z\text{-axis}$, we get
$$
\omega_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$
and it has 
$$
v_3 = -\omega_3 \times q_3 = -\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \times \begin{bmatrix} 3 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ -3 \\ 0 \end{bmatrix}
$$
which gets us the total transformation of
$$
\mathcal{S}_3 = \begin{bmatrix} \omega_3 \\ v_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ -3 \\ 0 \end{bmatrix}
$$
---
##### Finding $\theta_{2}$
For $\theta_{2}$, we have

![image|300](https://notes-media.kthiha.com/Product-of-Exponentials/2c75e71d8dc9d5e4f93b67838073c995.png)

Since this is a prismatic joint moving in the $x\text{-axis}$ direction, we get
$$
\mathcal{S}_2 = \begin{bmatrix} \omega_2 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{bmatrix}
$$

---
##### Finding $\theta_{1}$
For $\theta_{1}$, we have

![image|300](https://notes-media.kthiha.com/Product-of-Exponentials/b4e3c2db258eafcdb639c63ec85fb1dd.png)

Given $\theta_{1}$ is the revolute joint rotating in $z\text{-axis}$ with no distance from origin,
$$
\mathcal{S}_1 = \begin{bmatrix} \omega_1 \\ v_1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}
$$

---
#### Step-3: Forward Kinematics by Evaluating PoE
Given our transformations of
$$
\begin{array}{|c|c|c|}
\hline
\mathcal{S}_i & \omega_i & v_i \\
\hline
1 & (0,0,1) & (0,0,0) \\
2 & (0,0,0) & (1,0,0) \\
3 & (0,0,1) & (0,-3,0) \\
\hline
\end{array}

\qquad

M = \begin{bmatrix}
1 & 0 & 0 & 4 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}

\qquad

\theta = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \theta_3 \end{bmatrix} = \begin{bmatrix} \pi/6 \\ 0.5 \\ \pi/4 \end{bmatrix}
$$
we get the result of 
$$
\begin{align*}
T_{sb}(\theta) &= e^{[\mathcal{S}_1]\theta_1} e^{[\mathcal{S}_2]\theta_2} e^{[\mathcal{S}_3]\theta_3} M \\[1em]
&= \begin{bmatrix}
0.2588 & -0.9659 & 0 & 3.2899 \\
0.9659 & 0.2588 & 0 & 2.7159 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\end{align*}
$$

---
### Body Coordinate
#### Step-1: Determine the transformation $M$ matrix.
Given the following zero position of the robot

![image|400](https://notes-media.kthiha.com/Product-of-Exponentials/2ed91afc0c46a086c07a52a104a01a61.png)

We get the following matrices:
$$
\theta = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \theta_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
\qquad
M = \begin{bmatrix}
1 & 0 & 0 & 4 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

---
#### Step-2: Find screw axis $\mathcal{B}$ in $\{ b \}$-frame
##### Finding $\theta_{3}$
For $\theta_{3}$, we have

![image|300](https://notes-media.kthiha.com/Product-of-Exponentials/a248e02e6226371dd461745192df4454.png)

Given it is $4$ $x\text{-units}$ from the body $\{ b \}$ to the base in the direction of rotation of $z\text{-axis}$, 
$$
v_1 = -\omega_1 \times q_1 = -\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \times \begin{bmatrix} -4 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 4 \\ 0 \end{bmatrix}
$$
which gets us 
$$
\mathcal{B}_1 = \begin{bmatrix} \omega_1 \\ v_1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 4 \\ 0 \end{bmatrix}
$$

---
##### Finding $\theta_{2}$
For $\theta_{2}$, we have

![image|400](https://notes-media.kthiha.com/Product-of-Exponentials/2c75e71d8dc9d5e4f93b67838073c995.png)

Since this is a prismatic joint moving in the $x\text{-axis}$ direction, we get
$$
\mathcal{S}_2 = \begin{bmatrix} \omega_2 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{bmatrix}
$$

---
##### Finding $\theta_{1}$
For $\theta_{1}$, we have

![image|400](https://notes-media.kthiha.com/Product-of-Exponentials/84e35c3781a4807c2d844cbbcc9ffafd.png)

Given it is $1$ $x\text{-unit}$ from $\{ b \}$ with direction of rotation in $z\text{-axis}$ direction,
$$
v_3 = -\omega_3 \times q_3 = -\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \times \begin{bmatrix} -1 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
$$

which gets us
$$
\mathcal{B}_3 = \begin{bmatrix} \omega_3 \\ v_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 1 \\ 0 \end{bmatrix}
$$

---
#### Step-3: Forward Kinematics by Evaluating PoE
Given our transformations of
$$
\begin{array}{|c|c|c|} \hline \mathcal{B}_i & \omega_i & v_i \\ \hline 1 & (0,0,1) & (0,4,0) \\ 2 & (0,0,0) & (1,0,0) \\ 3 & (0,0,1) & (0,1,0) \\ \hline \end{array}

\qquad

M = \begin{bmatrix} 1 & 0 & 0 & 4 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}

\qquad

\theta = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \theta_3 \end{bmatrix} = \begin{bmatrix} \pi/6 \\ 0.5 \\ \pi/4 \end{bmatrix}
$$

we get the result of 

$$
\begin{align*}
T_{sb}(\theta) &= M e^{[\mathcal{B}_1]\theta_1} e^{[\mathcal{B}_2]\theta_2} e^{[\mathcal{B}_3]\theta_3} \\[1em]
&= \begin{bmatrix}
0.2588 & -0.9659 & 0 & 3.2899 \\
0.9659 & 0.2588 & 0 & 2.7159 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} \\[1em]
&= e^{[\mathcal{S}_1]\theta_1} e^{[\mathcal{S}_2]\theta_2} e^{[\mathcal{S}_3]\theta_3} M
\end{align*}
$$

---
## 6R Robot
Consider the following robot with $6$ revolute joints.

![image|400](https://notes-media.kthiha.com/Product-of-Exponentials/919290f29ea65ea76eef5237385341d3.png)

---
### Space Frame
We then have the transformations of 
$$
T(\theta) = M = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 3L \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
\qquad
\begin{array}{|c||c|c|} \hline i & \omega_i & v_i \\ \hline\hline 1 & (0,0,1) & (0,0,0) \\ 2 & (0,1,0) & (0,0,0) \\ 3 & (-1,0,0) & (0,0,0) \\ 4 & (-1,0,0) & (0,0,L) \\ 5 & (-1,0,0) & (0,0,2L) \\ 6 & (0,1,0) & (0,0,0) \\ \hline \end{array}
$$
Given the values of $\theta$, we can substitute it into
$$
\theta = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \theta_3 \\ \theta_4 \\ \theta_5 \\ \theta_6 \end{bmatrix}
\qquad
T_{sb}(\theta) = e^{[\mathcal{S}_1]\theta_1} e^{[\mathcal{S}_2]\theta_2} e^{[\mathcal{S}_3]\theta_3} e^{[\mathcal{S}_4]\theta_4} e^{[\mathcal{S}_5]\theta_5} e^{[\mathcal{S}_6]\theta_6} M
$$

---
### Body Frame
We then have the transformations of 
$$
T(\theta) = M = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 3L \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
\qquad
\begin{array}{|c||c|c|}
\hline
i & \omega_i & v_i \\
\hline\hline
1 & (0,0,1) & (-3L,0,0) \\
2 & (0,1,0) & (0,0,0) \\
3 & (-1,0,0) & (0,0,-3L) \\
4 & (-1,0,0) & (0,0,-2L) \\
5 & (-1,0,0) & (0,0,-L) \\
6 & (0,1,0) & (0,0,0) \\
\hline
\end{array}
$$

Given the values of $\theta$, we can substitute it into
$$
T(\theta) = M e^{[\mathcal{B}_1]\theta_1} e^{[\mathcal{B}_2]\theta_2} e^{[\mathcal{B}_3]\theta_3} e^{[\mathcal{B}_4]\theta_4} e^{[\mathcal{B}_5]\theta_5} e^{[\mathcal{B}_6]\theta_6}
$$

---
