# Analytical Inverse Kinematics
[[Analytical Inverse Kinematics|Analytical inverse kinematics]] mean solving for joint angles in closed form with no iteration, given a desired end-effector pose.

![300](https://la.mathworks.com/discovery/inverse-kinematics/_jcr_content/mainParsys/image.adapt.full.medium.jpg/1786447041984.jpg)

---
## Strategy
Given $X \in SE(3),$ find solutions $\theta \in \mathbb{R}^{n}$ satisfying $\text{FK}(\theta) = X$.

### When closed-form solution exists
> **Pieper's Criterion**
> A 6-DOF serial arm has a closed-form solution if three consecutive joint axes intersect at a point(eg: a spherical twist) or are parallel.

### Core Strategy (6-DOF)
For a 6-DOF arm with spherical wrist
- Compute the wrist center: $p_w = p_{target} − d_{6} · R_{target} · \hat{z}$ .
- Solve joints $1\text{-}3$ from $p_{w}$ (positioning from wrist center).
- Compute $R_{3}^{6} = (R_{0}^{3})^{T} \cdot R_{\text{target}}$ and extract joints $4\text{-}6$ as Euler angles.

---
## Example-1: Planar 2R Robot
Find the explicit solution $(\theta_{1} \text{ and } \theta_{2})$ to [[Inverse Kinematics|inverse kinematics]] for a desired end-effector position $(x,y)$.

![image|250](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/babd57d4f0f0ce2fa8f54b368b720996.png)

---
### Finding $\theta_{2}$ using $\beta$
We shall define $\beta$ and use it get an equation.

![image|300](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/f4a3de8e1a3b7930f0bea3edb5df0444.png)

Using the [[Law of Cosine|Law of Cosine]], we get
$$
x^2 + y^2 = L_1^2 + L_2^2 - 2L_1L_2\cos\beta
$$
Resolving for $\beta$, we get
$$
\beta = \cos^{-1}\left(\frac{L_1^2 + L_2^2 - x^2 - y^2}{2L_1L_2}\right)
$$
Determining $\theta_{2}$, we get
$$
\theta_{2} = \pi - \beta
$$

---
### Finding $\theta_{1}$ using $\alpha$
We shall define $\alpha$ and use it get an equation.

![image|300](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/7a41c7f8a20b6639d9d9d6c82959b3d5.png)

Using the [[Law of Cosine|Law of Cosine]], we get
$$
L_2^2 = L_1^2 + x^2 + y^2 - 2L_1\sqrt{x^2 + y^2}\cos\alpha
$$
Resolving for $\alpha$, we get
$$
\alpha = \cos^{-1}\left(\frac{x^2 + y^2 + L_1^2 - L_2^2}{2L_1\sqrt{x^2 + y^2}}\right)
$$
Using $\gamma=\operatorname{atan2}(y,x)$ to determine $\theta_{1},$ we get
$$
\theta_{1} = \gamma - \alpha
$$

---
### Elbow Down/Up Solution
For a given end-effector position $(x,y)$, we can get two different solutions.

![image|300](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/af44369fc07c7a5cdd060501052c7358.png)

Given the following
$$
\alpha = \cos^{-1}\left(\frac{x^2 + y^2 + L_1^2 - L_2^2}{2L_1\sqrt{x^2 + y^2}}\right) \qquad \beta = \cos^{-1}\left(\frac{L_1^2 + L_2^2 - x^2 - y^2}{2L_1L_2}\right) \qquad \gamma = \operatorname{atan2}(y, x)
$$

We can the `elbow down` solution of
$$
\theta_1 = \gamma - \alpha \qquad \theta_2 = \pi - \beta
$$
or the `elbow up` solution of
$$
\theta_1 = \gamma + \alpha \qquad \theta_2 = \beta - \pi
$$

---
### Final Result of Inverse Kinematics
As per the final result of the [[Inverse Kinematics|inverse kinematics]], we get

![image|200](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/68bd3f94bdcf8135f59e983212af0bd4.png)

- For $(x,y)$ in `interior` of the workspace, we get two solutions:

$$
\begin{array}{ll}
\text{elbow down} & \theta_1 = \gamma - \alpha \qquad \theta_2 = \pi - \beta \\[1em]
\text{elbow up} & \theta_1 = \gamma + \alpha \qquad \theta_2 = \beta - \pi
\end{array}
$$

- For $(x,y)$ on `boundary`, we get one solution:

$$
\begin{align*}
\theta_1 &= \operatorname{atan2}(y, x) \\
\theta_2 &= 0
\end{align*}
$$

- For $(x,y)$ in `exterior`, we have no solution:

$$
(x,y) \notin [L_{1} - L_{2}, \ L_{1} + L_{2}]
$$

---
## Example-2: 6R PUMA Robot
Find the explicit solution $(\theta_{1} \text{ and } \theta_{2})$ to [[Inverse Kinematics|inverse kinematics]] for a desired end-effector position $(x,y)$.

![image|250](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/4d8d79f0084c187d62bcdc9e11b5f227.png)

Given any $X = \begin{bmatrix} R & p \\ 0\;0\;0 & 1 \end{bmatrix} \in SE(3)$, we are decoupling into two sub-problems
- Position of wrist (first three joints)
- Orientation of wrist (last three joints)

So, we will be solving inverse-position $(\theta_{1}, \theta_{2}, \theta_{3})$ and inverse-orientation $(\theta_{4}, \theta_{5}, \theta_{6})$ subproblems.

---
### Inverse Position Subproblem (No Shoulder Offset)
We have the following robotic configuration.

![image|400](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/b3a58a0ab8b5993efa333a6e003f2313.png)

We can thus represent it as
$$
\begin{align*}
T(\theta) &= \begin{bmatrix} R & p \\ 0\;0\;0 & 1 \end{bmatrix} \\[1em]
p &= (p_x, p_y, p_z) \\[1em]
d_1 &= 0 \;\text{(offset to elbow)}
\end{align*}
$$
We can then use it to find $\theta_{1}$ as
$$
\theta_1 = \operatorname{atan2}(p_y, p_x) \qquad\qquad \theta_1 = \operatorname{atan2}(p_y, p_x) + \pi \;\; \text{if } \theta_2 = \pi - \theta_2
$$

---
#### Singular Configuration
Given $p_{x} = p_{y} = 0$, we can get infinite solutions for $\theta_{1}$.

![image|150](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/359417dcc89d1d6f6c5b360036559d11.png)

---
### Inverse Position Subproblem (With Shoulder Offset)
There are four possible inverse position problem solutions for shoulder offset.

![image|300](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/8e47d15d5fc785e775cd4a40692ea618.png)

---
#### Lefty
For the given robot configuration,

![image|300](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/abd407133d4fae138746fc81582d8e7c.png)

This gets us
![image|200](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/aeb6469bf9525c7fcefc9d74466c66a1.png)

where we have
$$
\theta_1 = \phi - \alpha \qquad\qquad r^2 = p_x^2 + p_y^2
$$
and we can get that
$$
\begin{align*}
\bbox[#F4CCCC,5px]{\color{#C00000}{\phi}} &= \operatorname{atan2}(p_y, p_x) \\[1em]
\bbox[#D9EAD3,5px]{\color{#2E7D32}{\alpha}} &= \operatorname{atan2}\left(d_1, \sqrt{r^2 - d_1^2}\right)
\end{align*}
$$

---
#### Righty
For the given robotic configuration, 

![image|200](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/2d8cfcfad5bcd661603183d861eb763d.png)

we can get that
![image|200](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/69a445404b9947dd4100d7ff6dcb6dcd.png)

$$
\theta_1 = \pi + \bbox[#F4CCCC,5px]{\color{#C00000}{\operatorname{atan2}(p_y, p_x)}} + \bbox[#D9EAD3,5px]{\color{#2E7D32}{\operatorname{atan2}\left(-\sqrt{r^2 - d_1^2}, \, d_1\right)}}
$$

---
#### Inverse Position Subproblem (with shoulder offset)
> We can reduce the problem to planar 2R [[Inverse Kinematics|inverse kinematics]] problem.

Recall that we have the following robotic configuration.

![image|400](https://notes-media.kthiha.com/Analytical-Inverse-Kinematics/b3a58a0ab8b5993efa333a6e003f2313.png)

We first get that
$$
\cos\theta_3 = \underbrace{\frac{r^2 - d_1^2 + p_z^2 - a_2^2 - a_3^2}{2a_2a_3}}_{= D}
$$
We can then use it to get
$$
\begin{align*}
\theta_3 &= \operatorname{atan2}\left(\pm\sqrt{1 - D^2}, \, D\right) \\[1em]
\theta_2 &= \operatorname{atan2}\left(p_z, \sqrt{r^2 - d_1^2}\right) - \operatorname{atan2}\left(a_3 \sin\theta_3, \, a_2 + a_3 \cos\theta_3\right)
\end{align*}
$$

---
### Inverse Orientation Subproblem
We can change the [[Product of Exponentials|product of exponentials]] equation to get
$$
\begin{align*}
X &= {\color{#00A651}e^{[\mathcal{S}_1]\theta_1} e^{[\mathcal{S}_2]\theta_2} e^{[\mathcal{S}_3]\theta_3}} \, e^{[\mathcal{S}_4]\theta_4} e^{[\mathcal{S}_5]\theta_5} e^{[\mathcal{S}_6]\theta_6} M \\[1em]
\Longleftrightarrow \quad e^{[\mathcal{S}_4]\theta_4} e^{[\mathcal{S}_5]\theta_5} e^{[\mathcal{S}_6]\theta_6} &= \underbrace{e^{-[\mathcal{S}_3]\theta_3} e^{-[\mathcal{S}_2]\theta_2} e^{-[\mathcal{S}_1]\theta_1} X M^{-1}}_{\color{gray}\text{Use } R \text{ from this homogeneous transformation matrix in the following}}
\end{align*}
$$
We than have that
$$
\mathcal{S}_4, \mathcal{S}_5, \mathcal{S}_6 \;\Longrightarrow\; \omega_4 = (0,0,1), \; \omega_5 = (0,1,0), \; \omega_6 = (1,0,0)
$$
Hence, we need to solve for
$$
R = \mathrm{Rot}(\hat{z}, \theta_4)\,\mathrm{Rot}(\hat{y}, \theta_5)\,\mathrm{Rot}(\hat{x}, \theta_6)
$$

---
#### Solving for R
In order to solve for $R = \mathrm{Rot}(\hat{z}, \theta_4)\,\mathrm{Rot}(\hat{y}, \theta_5)\,\mathrm{Rot}(\hat{x}, \theta_6),C$ we first get that
$$
R = \begin{bmatrix}
c_4 c_5 & c_4 s_5 s_6 - s_4 c_6 & c_4 s_5 c_6 + s_4 s_6 \\
s_4 c_5 & s_4 s_5 s_6 + c_4 c_6 & s_4 s_5 c_6 - c_4 s_6 \\
-s_5 & c_5 s_6 & c_5 c_6
\end{bmatrix}
\qquad c_n = \cos\theta_n, \;\; s_n = \sin\theta_n
$$

> **Case-1**: $r_{31} \neq \pm 1$

$$
\begin{align*}
\theta_4 &= \operatorname{atan2}(r_{21}, r_{11}) \\[1em]
\theta_5 &= \operatorname{atan2}\left(-r_{31}, \sqrt{r_{11}^2 + r_{21}^2}\right) \\[1em]
\theta_6 &= \operatorname{atan2}(r_{32}, r_{33})
\end{align*}
$$

> **Case-2**: $r_{31} = -s_{5} = -1$ and $c_{5 = 0}$

$$
\theta_4 = 0 \qquad \theta_5 = \frac{\pi}{2} \qquad \theta_6 = \operatorname{atan2}(r_{12}, r_{22})
$$

> **Case-3**: $r_{31} = -s_{5} = 1$ and $c_{5} = 0$

$$
\theta_4 = 0 \qquad \theta_5 = -\frac{\pi}{2} \qquad \theta_6 = -\operatorname{atan2}(r_{12}, r_{22})
$$

---
