# Twist 
[[Twist|Twist]] in robotics is used to refer to a [[Velocity Kinematics|spatial velocity]].

![image|200](https://notes-media.kthiha.com/Velocity-Kinematics/665c724e285dcf2c082a26b0d67e43bf.png)

---
### Linear Velocity
> The [[#Linear Velocity|linear velocity]] $v$ of a [[Rigid Body|rigid body]] is a velocity quantity equal to the time rate of change of its linear position. 

$$
\begin{align*}
v &= \frac{s}{t} \qquad \frac{\text{distance}}{\text{time}} \\[1em]
s &= vt
\end{align*}
$$
> Its essentially the velocity of a reference point fixed to the body.

---
### Angular Velocity
> The [[#Angular Velocity|angular velocity]] $\omega$ of a [[Rigid Body|rigid body]] is a vector quantity equal to its angular displacement per unit of time. 

$$
\begin{align*}
\omega &= \frac{\theta}{t} \qquad \frac{\text{angular displacement}}{\text{time}} \\[1em]
\theta &= \omega t
\end{align*}
$$

---
## Twist
A [[Twist|6-vector twist]] can be used to represent a [[Rigid Body|rigid body velocity]].

Multiplying our representation of the [[Screw Axis|screw axis]] $\mathcal{S}$ by the scalar rate of rotation $\dot{\theta}$, 
$$
\mathcal{V} = \mathcal{S}\dot{\theta} \qquad \mathcal{S} = \begin{bmatrix} \omega \\ v \end{bmatrix}
$$

$(i)$ If pitch $h$ is infinite: 
$$
\omega = 0, \; \|v\| = 1, \; \dot{\theta} = \text{linear speed}
$$
$(ii)$ If pitch $h$ is finite:
$$
\|\omega\| = 1, \; \dot{\theta} = \text{rotational speed}
$$

---
## Spatial Twist and Body Twist
> If the [[Screw Axis|screw axis]] $\mathcal{B}$ is defined in $\{ b \}$, $\mathcal{V}_{b} = (\omega_{b}, \ v_{b}) = \mathcal{B}\dot{\theta}$ is the [[Twist|body twist]].
> If the [[Screw Axis|screw axis]] $\mathcal{S}$ is defined in $\{ s \}$, $\mathcal{V}_{s} = (\omega_{s}, \ v_{s}) = \mathcal{S}\dot{\theta}$ is the [[Twist|spatial twist]].

Both the [[Twist|body twist]] and [[Twist|spatial twist]] represent the same motion, but with respect to different coordinate frames.
- The [[Twist|body twist]] is not affected by the choice of the space frame.
- The [[Twist|spatial twist]] is not affected by the choice of the body frame.

---
## Matrix Representation of a Twist
Given a fixed (space) frame $\{s\}$, a body frame $\{b\}$, and a differentiable $T_{sb}(t) \in SE(3)$, where
$$
T_{sb}(t) = \begin{bmatrix} R(t) & p(t) \\ 0 & 1 \end{bmatrix}
$$
then the matrix representation of the [[Twist|body twist]] is
$$
T_{sb}^{-1}\dot{T}_{sb} = [\mathcal{V}_b] = \begin{bmatrix} [\omega_b] & v_b \\ 0 & 0 \end{bmatrix} \in se(3)
$$
and the matrix representation of the [[Twist|spatial twist]] is
$$
\dot{T}_{sb}T_{sb}^{-1} = [\mathcal{V}_s] = \begin{bmatrix} [\omega_s] & v_s \\ 0 & 0 \end{bmatrix} \in se(3)
$$

---
### Relation between $\mathcal{V}_{S}$ and $\mathcal{V}_{b}$
Note that given $T = (R, p) \in SE(3)$, its adjoint representation is
$$
[Ad_T] = \begin{bmatrix} R & 0 \\ [p]R & R \end{bmatrix} \in \mathbb{R}^{6\times 6}
$$
The twists $\mathcal{V}_s$ and $\mathcal{V}_b$ are related by
$$
\begin{align*}
\mathcal{V}_s = \begin{bmatrix} \omega_s \\ v_s \end{bmatrix} &= \begin{bmatrix} R & 0 \\ [p]R & R \end{bmatrix} \begin{bmatrix} \omega_b \\ v_b \end{bmatrix} = [Ad_{T_{sb}}]\mathcal{V}_b \\[1em]
\mathcal{V}_b = \begin{bmatrix} \omega_b \\ v_b \end{bmatrix} &= \begin{bmatrix} R^T & 0 \\ -R^T[p] & R^T \end{bmatrix} \begin{bmatrix} \omega_s \\ v_s \end{bmatrix} = [Ad_{T_{bs}}]\mathcal{V}_s
\end{align*}
$$

---
