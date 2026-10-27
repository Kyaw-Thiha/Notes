# Robot Jacobian
[[Robot Jacobian|Robot Jacobian]] can be used to find what different joint angle velocities will cause in terms of end-effector [[Velocity Kinematics|linear]] and [[Velocity Kinematics|angular velocities]].

---
## Definition 
### Space Jacobian
Let the [[Forward Kinematics|forward kinematics]] of a $n\text{-link}$ serial robot arm be expressed in the following [[Product of Exponentials|product of exponentials]] form 
$$
T_{sb}(\theta) = e^{[S_{1}]\theta_{1}} \ e^{[S_{2}]\theta_{2}} \dots e^{[S_{n}]\theta_{n}} M
$$
> The [[Robot Jacobian|space Jacobian]] $J_{S}(\theta) \in \mathbb{R}^{6\times n}$ relates the joint rate vector $\dot{\theta} \in \mathbb{R}^{n}$ to the [[Twist|spatial twist]] $\mathcal{V}_{S}$ via $\mathcal{V}_{S}=J_{S}(\theta) \ \dot{\theta}$ with 
$$
J_{S}(\theta)
= [J_{S1} \quad J_{S 2}(\theta) \quad \dots \quad J_{S n}(\theta)] \in 
\mathbb{R}^{6\times n}
$$
> where the $i^{th}$ column is given by 
$$
J_{si}(\theta) = \mathrm{Ad}_{e^{[\mathcal{S}_1]\theta_1} \cdots e^{[\mathcal{S}_{i-1}]\theta_{i-1}}}(\mathcal{S}_i)
$$
> for $i=2, \dots, n$ with the first column $J_{s 1} = S_{1}$.

The screw vector $J_{s i}$ for joint $i$ is expressed in space-frame coordinates, with the joint values $\theta$ assumed to be arbitrary rather than zero.

---
### Body Jacobian
Let the forward kinematics of an $n\text{-link}$ serial robot arm be expressed in the following product of exponentials form

$$
T_{sb}(\theta) = M \, e^{[\mathcal{B}_{1}]\theta_{1}} \, e^{[\mathcal{B}_{2}]\theta_{2}} \dots e^{[\mathcal{B}_{n}]\theta_{n}}
$$

> The body Jacobian $J_{b}(\theta) \in \mathbb{R}^{6\times n}$ relates the joint rate vector $\dot{\theta} \in \mathbb{R}^{n}$ to the end-effector body twist $\mathcal{V}_{b}$ via $\mathcal{V}_{b}=J_{b}(\theta) \ \dot{\theta}$ with

$$
J_{b}(\theta) = [J_{b1}(\theta) \quad \dots \quad J_{b,n-1}(\theta) \quad J_{bn}] \in \mathbb{R}^{6\times n}
$$

> where the $i^{th}$ column is given by

$$
J_{bi}(\theta) = \mathrm{Ad}_{e^{-[\mathcal{B}_n]\theta_n} \cdots e^{-[\mathcal{B}_{i+1}]\theta_{i+1}}}(\mathcal{B}_i)
$$

> for $i=n-1, \dots, 1$ with the last column $J_{bn} = \mathcal{B}_{n}$.

The screw vector $J_{bi}$ for joint $i$ is expressed in body-frame coordinates, with the joint values $\theta$ assumed to be arbitrary rather than zero.

---
## Procedure
> Derive the [[Velocity Kinematics|velocity kinematics]] ([[Robot Jacobian|space Jacobian]]) of a robot.

> Determine each column $J_{s i}(\theta)$ as the [[Screw Axis|screw]] describing joint axis $i$, expressed in fixed-frame coordinates for arbitrary $\theta$ (rather than $\theta=0$ as in [[Product of Exponentials|FK PoE]]).

> Compute each column using Adjoint $J_{si}(\theta) = \mathrm{Ad}_{e^{[\mathcal{S}_1]\theta_1} \cdots e^{[\mathcal{S}_{i-1}]\theta_{i-1}}}(\mathcal{S}_i)$ 
> or
> Determine each $J_{s i}(\theta)$ implicitly(manually).

---
## Example-1
Suppose we have the following $2R$ robot.

![image|400](https://notes-media.kthiha.com/Robot-Jacobian/04afc090690b993a4ed68d9543a9fb0e.png)

---
### Step-1: Determine $S_{i}$
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

---
### Step-2: Compute columns of $J_{S}(\theta)$
Given the angles
$$
\theta = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \theta_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 45^\circ \\ 30^\circ \end{bmatrix}
$$
We can compute 
$$
\begin{align*}
J_s(\theta) &= \begin{bmatrix} J_{s1} & J_{s2}(\theta) & \cdots & J_{sn}(\theta) \end{bmatrix} \in \mathbb{R}^{6\times 3} \\[1em]
J_{si}(\theta) &= \mathrm{Ad}_{e^{[\mathcal{S}_1]\theta_1} \cdots e^{[\mathcal{S}_{i-1}]\theta_{i-1}}}(\mathcal{S}_i)
\end{align*}
$$
and get that
$$
J_s(\theta) = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
1 & 1 & 1 \\
0 & 0 & 1.414 \\
0 & -2 & -3.414 \\
0 & 0 & 0
\end{bmatrix}
$$

---
### Step-3: Given joint velocities $\dot{\theta}$ find $v_{S}$
The angles $\theta$ lead to the robot in the configuration as shown below.

![image|300](https://notes-media.kthiha.com/Robot-Jacobian/5598f2d141bd9de02244f0ac63cfe287.png)

Given [[Velocity Kinematics|joint velocities]] $\dot{\theta}$, find $v_{S}$,
$$
\dot{\theta} = \begin{bmatrix} 1^\circ \\ 1^\circ \\ 1^\circ \end{bmatrix} \qquad v_s = J_s(\theta)\dot{\theta} = \begin{bmatrix} 0 \\ 0 \\ 3 \\ 1.414 \\ -5.414 \\ 0 \end{bmatrix}
$$

---
## Example-2
Determine the [[Robot Jacobian|space Jacobian]] for a spatial RRRP chain implicitly. The robot is in $\theta = [0, \ \frac{\pi}{2}, 0, 0 ]^{T}$.

![image|300](https://notes-media.kthiha.com/Robot-Jacobian/151e342a1b607b7ebe46e90ad454b476.png)

We then get the [[Robot Jacobian|robot jacobian]] of

![image|400](https://notes-media.kthiha.com/Robot-Jacobian/531042b4e780ebd2ae9b9f156006a00e.png)

---
## Geometric Jacobian 
> The [[Velocity Kinematics|end-effector velocity]] is represented by [[Twist|twist]] $\mathcal{V}$. 
> $J_{s}(\theta)$ and $J_{b}(\theta)$ are called [[#Geometric Jacobian|geometric Jacobians]].

$$
\begin{array}{l c l}
[\mathcal{V}_s] = \dot{T}_{sb}T_{sb}^{-1} & \hspace{4em} & [\mathcal{V}_b] = T_{sb}^{-1}\dot{T}_{sb} \\[1em]
\mathcal{V}_s = \mathrm{Ad}_{T_{sb}}(\mathcal{V}_b) = J_s(\theta)\dot{\theta} & & \mathcal{V}_b = \mathrm{Ad}_{T_{bs}}(\mathcal{V}_s) = J_b(\theta)\dot{\theta} \\[1em]
J_s(\theta) = \mathrm{Ad}_{T_{sb}}\big(J_b(\theta)\big) & & J_b(\theta) = \mathrm{Ad}_{T_{bs}}\big(J_s(\theta)\big)
\end{array}
$$

---
## Analytic Jacobian
> End-effector configuration $q$ with minimum set of coordinates.
> $\text{E.g:}$ [[Euler Rotation|Euler angles]] $(\text{roll, pitch, yaw})$ and Cartesian coordinates $(x,y,z)$.

> [[Forward Kinematics|Forward kinematics]] map joint values to minimal coordinate representation.

$$
\begin{bmatrix} q_1 \\ \vdots \\ q_m \end{bmatrix} = \begin{bmatrix} f_1(\theta) \\ \vdots \\ f_m(\theta) \end{bmatrix} \in \mathbb{R}^m
$$
> [[Velocity Kinematics|End-effector velocity]] is given the time derivative of the coordinates $\dot{q} = J_{a}(\theta) \ \dot{\theta}$, where $J_{a}(\theta)$ is called [[#Analytic Jacobian|analytical Jacobian]]. 

$$
J_a(\theta) = \frac{\partial f(\theta)}{\partial \theta} = \begin{bmatrix}
\dfrac{\partial f_1}{\partial \theta_1} & \cdots & \dfrac{\partial f_1}{\partial \theta_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial f_m}{\partial \theta_1} & \cdots & \dfrac{\partial f_m}{\partial \theta_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

---
